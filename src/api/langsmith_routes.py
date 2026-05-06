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

# LangSmith organization and project IDs for constructing private trace URLs
# These are used to build URLs like: https://smith.langchain.com/o/{org_id}/projects/p/{project_id}?peek={run_id}
LANGSMITH_ORG_ID = os.getenv("LANGSMITH_ORG_ID")
LANGSMITH_PROJECT_ID = os.getenv("LANGSMITH_PROJECT_ID")

# Deployment environment: 'internal' uses private URLs, 'external' uses public share URLs
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "external")

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
    """Generate a trace URL for a LangSmith run.

    Behavior depends on DEPLOYMENT_ENV:
    - 'internal': Creates a private trace URL (requires LangSmith authentication)
      - Keeps sensitive data secure by NOT creating public share links
      - Requires LANGSMITH_ORG_ID environment variable
    - 'external': Creates a public share URL via LangSmith API
      - Anyone with the link can view the trace (no authentication required)
      - Good for public-facing deployments

    Environment variables:
    - DEPLOYMENT_ENV: 'internal' or 'external' (default: 'external')
    - LANGSMITH_ORG_ID: Required for internal deployments
    - LANGSMITH_PROJECT_ID: Required for internal deployments (UUID of your LangSmith project)
    """
    # External deployment: use public share URLs (default behavior)
    if DEPLOYMENT_ENV == "external":
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

    # Internal deployment: construct private trace URL
    if not LANGSMITH_ORG_ID:
        raise HTTPException(
            status_code=500,
            detail="LANGSMITH_ORG_ID environment variable required for internal deployments",
        )

    # Construct private trace URL
    # Format: https://smith.langchain.com/o/{org_id}/projects/p/{project_id}?peek={run_id}&peeked_trace={run_id}
    base_url = "https://smith.langchain.com"

    if LANGSMITH_PROJECT_ID:
        # Link to specific project
        trace_url = f"{base_url}/o/{LANGSMITH_ORG_ID}/projects/p/{LANGSMITH_PROJECT_ID}?peek={runId}&peeked_trace={runId}"
    else:
        # Fallback: Link to organization
        trace_url = f"{base_url}/o/{LANGSMITH_ORG_ID}?peek={runId}&peeked_trace={runId}"

    return {"shareUrl": trace_url}
