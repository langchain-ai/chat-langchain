# FastAPI server for public Chat LangChain support endpoints
import logging
import os
import re
import string
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.langsmith_routes import router as langsmith_router
from src.tools.docs_tools import clear_cache, get_cache_stats

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://smith.langchain.com",
    "https://chat.langchain.com",
    "https://reference.langchain.com",
    "https://chat-lang-chain-v2.vercel.app",
    "https://chat-langchain-alpha.vercel.app",
    "https://public-chat-langchain-test.vercel.app",
    "https://public-chat-langchain-test-b5cwr3ocz-langchain.vercel.app",
]


def _get_cors_origins() -> list[str]:
    """Get CORS allowed origins from defaults plus environment overrides."""
    origins = DEFAULT_CORS_ORIGINS.copy()
    additional = os.getenv("ALLOWED_ORIGINS", "")
    if additional:
        origins.extend([o.strip() for o in additional.split(",") if o.strip()])
    return origins


app = FastAPI(
    title="Chat LangChain API Server",
    description="Public Chat LangChain support endpoints",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(langsmith_router)


class TitleGenerationRequest(BaseModel):
    """Request model for title generation."""

    userMessage: str
    assistantResponse: Optional[str] = None
    maxLength: Optional[int] = 60


class TitleGenerationResponse(BaseModel):
    """Response model for title generation."""

    title: str


def truncate_title(message: str, max_length: int = 60) -> str:
    """Generate a deterministic fallback conversation title."""
    title = message.strip()
    title = re.sub(
        r"^(how do i|how to|can you|please|help me with|i need help with)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = title.rstrip(string.punctuation)
    if title:
        title = title[0].upper() + title[1:]
    if len(title) > max_length:
        title = title[: max_length - 3] + "..."
    return title


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "chat-langchain"}


@app.post("/generate-title", response_model=TitleGenerationResponse)
async def generate_conversation_title(request: TitleGenerationRequest):
    """Generate a simple conversation title for the frontend."""
    return TitleGenerationResponse(
        title=truncate_title(request.userMessage, request.maxLength or 60)
    )


@app.get("/metrics/cache")
async def get_cache_metrics():
    """Get documentation search cache statistics."""
    stats = get_cache_stats()
    total_hits = stats["hits_exact"] + stats["hits_fuzzy"]
    total_requests = stats["total_requests"]
    stats["api_calls_saved"] = total_hits
    stats["api_calls_made"] = stats["misses"]
    stats["cost_reduction_percent"] = (
        round((total_hits / total_requests) * 100, 1) if total_requests else 0.0
    )

    return {
        "status": "success",
        "cache_metrics": stats,
        "description": "Mintlify API documentation search cache statistics",
    }


@app.post("/metrics/cache/clear")
async def clear_cache_endpoint():
    """Clear the documentation search cache and reset metrics."""
    stats_before = get_cache_stats()
    old_entries = stats_before.get("total_entries", 0)
    clear_cache()

    logger.warning("Cache manually cleared via API: %s entries removed", old_entries)
    return {
        "status": "success",
        "message": "Cache cleared successfully",
        "entries_removed": old_entries,
        "cache_stats": get_cache_stats(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chat LangChain API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_title": "/generate-title",
            "cache_metrics": "/metrics/cache",
            "cache_clear": "/metrics/cache/clear (POST)",
            "langsmith": "/langsmith",
        },
    }
