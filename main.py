"""
Article Checker — Backend
- ChromaDB for persistent vector storage
- Ollama for embeddings (nomic-embed-text) and reasoning (qwen3)
- Streaming SSE for real-time fact-check results
- Server-side PDF/DOCX/TXT extraction
- Auto-opens browser on startup
"""

import asyncio
import importlib
import io
import json
import re
import threading
import traceback
import uuid
import webbrowser
from pathlib import Path
from typing import Optional, AsyncGenerator

import chromadb
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────

OLLAMA_BASE     = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"
DEFAULT_MODEL   = "qwen3:8b"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "articles"
TOP_K           = 8
OLLAMA_TIMEOUT  = 180
PORT            = 8000

# ─── CHROMA ───────────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Article Checker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ─── TEXT EXTRACTION ──────────────────────────────────────────────────────────

def extract_text_from_file(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        return data.decode("utf-8", errors="replace")

    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ValueError("python-docx not installed. Run: pip install python-docx")
        doc = docx.Document(io.BytesIO(data))
        parts = []
        for p in doc.paragraphs:
            if p.text.strip():
                parts.append(p.text.strip())
        if not parts:
            raise ValueError("DOCX appears to be empty or has no readable paragraphs.")
        return "\n".join(parts)

    if ext == ".pdf":
        # pymupdf can be imported as either 'fitz' or 'pymupdf' depending on version
        fitz = None
        for mod_name in ("fitz", "pymupdf"):
            try:
                fitz = importlib.import_module(mod_name)
                break
            except ImportError:
                continue
        if fitz is None:
            raise ValueError("pymupdf not installed. Run: pip install pymupdf")
        pdf = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in pdf]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("PDF appears to have no extractable text (may be scanned/image-only).")
        return text

    raise ValueError(f"Unsupported file type: '{ext}'. Accepted: .txt, .pdf, .docx")


# ─── SENTENCE SPLITTING ───────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 15]


# ─── OLLAMA ───────────────────────────────────────────────────────────────────

async def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        for text in texts:
            # Truncate to 500 words — stays well within nomic-embed-text's 2048 token limit
            truncated = " ".join(text.split()[:500])
            last_err = None
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{OLLAMA_BASE}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": truncated}
                    )
                    resp.raise_for_status()
                    vectors.append(resp.json()["embedding"])
                    break
                except httpx.HTTPStatusError as e:
                    last_err = e
                    await asyncio.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s
            else:
                raise last_err
    return vectors


async def ollama_reason(input_sentence: str, candidates: list[dict], model: str) -> dict:
    if not candidates:
        return {"verdict": "NO_MATCH", "source": None, "match": None, "reason": None}

    candidate_block = "\n".join(
        f'[{i+1}] (Source: "{c["article"]}") {c["sentence"]}'
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a fact-checking assistant. Determine whether the INPUT SENTENCE is supported or contradicted by the SOURCE SENTENCES.

INPUT SENTENCE:
"{input_sentence}"

SOURCE SENTENCES:
{candidate_block}

Rules:
- If a source sentence clearly supports the input, respond: VERDICT: SUPPORTS
- If a source sentence clearly contradicts the input, respond: VERDICT: CONTRADICTS
- If no source is relevant enough, respond: VERDICT: NO_MATCH
- Then: SOURCE: <article name>
- Then: MATCH: <the exact source sentence that matched>
- Then: REASON: <one sentence explanation>
- If NO_MATCH: SOURCE: none  MATCH: none  REASON: none

Respond with ONLY these four lines. No preamble, no thinking tags, no extra text."""

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1}}
        )
        resp.raise_for_status()

    raw = resp.json().get("response", "")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    def extract(pattern):
        m = re.search(pattern, raw, re.IGNORECASE)
        return m.group(1).strip() if m else None

    verdict_raw = extract(r"VERDICT:\s*(SUPPORTS|CONTRADICTS|NO_MATCH)")
    verdict     = (verdict_raw or "NO_MATCH").upper()
    source      = extract(r"SOURCE:\s*(.+)")
    match_      = extract(r"MATCH:\s*(.+)")
    reason      = extract(r"REASON:\s*(.+)")

    return {
        "verdict": verdict,
        "source":  None if source  in (None, "none") else source,
        "match":   None if match_  in (None, "none") else match_,
        "reason":  None if reason  in (None, "none") else reason,
    }


# ─── ROUTES: LIBRARY ──────────────────────────────────────────────────────────

@app.get("/articles")
async def list_articles():
    result = collection.get(include=["metadatas"])
    articles: dict[str, dict] = {}
    for meta in result["metadatas"]:
        name = meta["article"]
        if name not in articles:
            articles[name] = {"sentences": 0}
        articles[name]["sentences"] += 1
    return [{"name": k, "sentences": v["sentences"]} for k, v in sorted(articles.items())]


@app.get("/articles/{name}/text")
async def get_article_text(name: str):
    result = collection.get(where={"article": name}, include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(404, f"Article '{name}' not found.")
    sentences = [m["sentence"] for m in result["metadatas"]]
    return {"name": name, "text": " ".join(sentences)}


async def _store_article(name: str, text: str):
    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(400, "No usable sentences found.")
    existing = collection.get(where={"article": name}, include=["metadatas"])
    if existing["ids"]:
        raise HTTPException(409, f"Article '{name}' already exists. Delete it first.")
    try:
        vectors = await embed(sentences)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Ollama embedding error: {e}")
    ids       = [str(uuid.uuid4()) for _ in sentences]
    metadatas = [{"article": name, "sentence": s} for s in sentences]
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
    return len(sentences)


@app.post("/articles")
async def add_article(name: str = Form(...), text: str = Form(...)):
    count = await _store_article(name, text)
    return {"added": count, "article": name}


@app.post("/articles/upload")
async def upload_article(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    data = await file.read()
    try:
        text = extract_text_from_file(file.filename, data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Extraction failed: {e}")
    article_name = (name or "").strip() or Path(file.filename).stem
    count = await _store_article(article_name, text)
    return {"added": count, "article": article_name}


@app.delete("/articles/{name}")
async def delete_article(name: str):
    existing = collection.get(where={"article": name}, include=["metadatas"])
    if not existing["ids"]:
        raise HTTPException(404, f"Article '{name}' not found.")
    collection.delete(ids=existing["ids"])
    return {"deleted": len(existing["ids"]), "article": name}


# ─── ROUTES: FACT-CHECK (STREAMING) ───────────────────────────────────────────

class CheckRequest(BaseModel):
    text: str
    top_k: int = TOP_K
    reason_model: str = DEFAULT_MODEL


async def stream_check(sentences: list[str], top_k: int, model: str) -> AsyncGenerator[str, None]:
    total = collection.count()

    for i, sentence in enumerate(sentences):
        # progress ping
        yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': len(sentences), 'sentence': sentence})}\n\n"

        try:
            q_vec = (await embed([sentence]))[0]
        except Exception as e:
            yield f"data: {json.dumps({'type': 'result', 'index': i, 'sentence': sentence, 'error': str(e)})}\n\n"
            continue

        k = min(top_k, total)
        hits = collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["metadatas", "distances"]
        )
        candidates = [
            {"sentence": m["sentence"], "article": m["article"], "distance": d}
            for m, d in zip(hits["metadatas"][0], hits["distances"][0])
            if d < 0.9
        ]

        try:
            result = await ollama_reason(sentence, candidates, model)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'result', 'index': i, 'sentence': sentence, 'error': str(e)})}\n\n"
            continue

        yield f"data: {json.dumps({'type': 'result', 'index': i, 'sentence': sentence, **result})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/check/stream")
async def check_stream(req: CheckRequest):
    sentences = split_sentences(req.text)
    if not sentences:
        raise HTTPException(400, "No usable sentences found.")
    if collection.count() == 0:
        raise HTTPException(400, "Library is empty.")
    return StreamingResponse(
        stream_check(sentences, req.top_k, req.reason_model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract plain text from uploaded file for the fact-check input panel."""
    data = await file.read()
    try:
        text = extract_text_from_file(file.filename, data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Extraction failed: {e}")
    return {"text": text, "filename": file.filename}


# ─── ROUTES: STATUS ───────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    ollama_ok = False
    models = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            if r.status_code == 200:
                ollama_ok = True
                models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return {
        "ollama": ollama_ok,
        "sentences_stored": collection.count(),
        "models": models
    }


# ─── STATIC (frontend) ────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────

def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    threading.Timer(1.2, open_browser).start()
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)