import logging
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager

from models import IngestRequest, ChatRequest, ChatResponse
from rag_service import RAGService

logging.basicConfig(level=logging.INFO)
rag = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("RAG Agent starting...")
    yield
    print("RAG Agent shutting down...")

app = FastAPI(title="Capstone 1 - RAG Agent", lifespan=lifespan)

@app.post("/ingest")
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    bg.add_task(rag.ingest, req.files_directory)
    return {"status": "Ingestion started"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    answer, sources, mode, latency = rag.generate_answer(req.query)
    return ChatResponse(
        answer=answer,
        sources=sources,
        mode=mode,
        latency_seconds=round(latency, 2)
    )
