# LangSmith API proxy routes for FastAPI app
import asyncio
import logging
import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from langsmith import Client as LangSmithClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/langsmith", tags=["langsmith"])

# Use custom env var name to avoid conflicts with LangGraph Cloud reserved names
# In LangGraph Cloud: set CHAT_LANGCHAIN_LANGSMITH_API_KEY instead of LANGSMITH_API_KEY
LANGSMITH_API_KEY = os.getenv("CHAT_LANGCHAIN_LANGSMITH_API_KEY") or os.getenv(
    "LANGSMITH_API_KEY"
)
LANGSMITH_BASE_URL = os.getenv("LANGSMITH_BASE_URL", "https://api.smith.langchain.com")

# Demo workspace API key — feedback is mirrored here alongside the primary workspace
LANGSMITH_DEMO_API_KEY = os.getenv("LANGSMITH_DEMO_API_KEY", "")

# Primary client (singleton)
_langsmith_client: Optional[LangSmithClient] = None

# Demo workspace client for feedback fan-out (singleton)
_demo_client: Optional[LangSmithClient] = None


def get_langsmith_client() -> LangSmithClient:
    """Get or create primary LangSmith client instance (singleton)."""
    global _langsmith_client

    if _langsmith_client is None:
        try:
            if LANGSMITH_API_KEY:
                _langsmith_client = LangSmithClient(
                    api_key=LANGSMITH_API_KEY, api_url=LANGSMITH_BASE_URL
                )
            else:
                _langsmith_client = LangSmithClient(api_url=LANGSMITH_BASE_URL)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize LangSmith client: {e}"
            )

    return _langsmith_client


def get_demo_client() -> Optional[LangSmithClient]:
    """Get demo workspace LangSmith client. Returns None if not configured."""
    global _demo_client

    if _demo_client is None and LANGSMITH_DEMO_API_KEY:
        try:
            _demo_client = LangSmithClient(
                api_key=LANGSMITH_DEMO_API_KEY, api_url=LANGSMITH_BASE_URL
            )
        except Exception as e:
            logger.warning(f"Failed to init demo LangSmith client: {e}")

    return _demo_client


def score_to_float(score: str) -> float:
    """Convert score string to float (reusable helper)."""
    return 1.0 if score == "positive" else 0.0


class FeedbackRequest(BaseModel):
    """Request model for feedback operations."""

    runId: str
    feedbackKey: str
    score: str  # "positive" or "negative"
    comment: Optional[str] = None
    feedbackId: Optional[str] = None


async def _mirror_to_demo(fn_name: str, *args, **kwargs):
    """Best-effort: mirror a feedback call to the demo workspace. Failures are logged, not raised."""
    client = get_demo_client()
    if client is None:
        return
    try:
        await asyncio.to_thread(getattr(client, fn_name), *args, **kwargs)
    except Exception as e:
        logger.warning(f"Demo workspace {fn_name} failed: {e}")


@router.post("/feedback")
async def create_or_update_feedback(request: FeedbackRequest):
    """Create or update feedback on a LangSmith run."""
    client = get_langsmith_client()
    score_value = score_to_float(request.score)

    # Try to update existing feedback if feedbackId provided
    if request.feedbackId:
        try:
            await asyncio.to_thread(
                client.update_feedback,
                request.feedbackId,
                score=score_value,
                comment=request.comment,
            )
            # Fire-and-forget to secondary workspaces (they create since they won't share feedback IDs)
            asyncio.create_task(
                _mirror_to_demo(
                    "create_feedback",
                    request.runId,
                    request.feedbackKey,
                    score=score_value,
                    comment=request.comment,
                )
            )
            return {"id": request.feedbackId}
        except Exception as e:
            # If feedback not found (404), fall through to create new one
            if hasattr(e, "status_code") and e.status_code == 404:
                pass
            else:
                raise HTTPException(status_code=500, detail=str(e))

    # Create new feedback
    feedback = await asyncio.to_thread(
        client.create_feedback,
        request.runId,
        request.feedbackKey,
        score=score_value,
        comment=request.comment,
    )
    # Fire-and-forget to secondary workspaces
    asyncio.create_task(
        _mirror_to_demo(
            "create_feedback",
            request.runId,
            request.feedbackKey,
            score=score_value,
            comment=request.comment,
        )
    )
    return {"id": feedback.id}


@router.delete("/feedback")
async def delete_feedback(
    feedbackId: str = Query(..., description="Feedback ID to delete"),
):
    """Delete feedback from LangSmith."""
    client = get_langsmith_client()
    await asyncio.to_thread(client.delete_feedback, feedbackId)
    # Note: secondary workspaces created their own feedback IDs, so we can't delete by the primary's ID.
    # Secondary feedback is orphaned on delete — acceptable tradeoff for simplicity.
    return {"success": True}


@router.get("/runs/{runId}")
async def read_run(runId: str):
    """Read run details from LangSmith."""
    try:
        client = get_langsmith_client()
        run = await asyncio.to_thread(client.read_run, runId)
        return run
    except Exception as e:
        # Return 404 for not found, let CORS middleware handle headers
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(
                status_code=404, detail="Run not found yet - may still be processing"
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{runId}/share")
async def share_run(runId: str):
    """Create a public share URL for a LangSmith run."""
    try:
        client = get_langsmith_client()
        share_url = await asyncio.to_thread(client.share_run, runId)
        return {"shareUrl": share_url}
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(
                status_code=404,
                detail="Run not found yet - may still be processing",
            )
        raise HTTPException(status_code=500, detail=str(e))
