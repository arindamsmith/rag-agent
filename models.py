from typing import List, Optional
from pydantic import BaseModel

class IngestRequest(BaseModel):
    files_directory: str = "data"

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    mode: str
    latency_seconds: float
