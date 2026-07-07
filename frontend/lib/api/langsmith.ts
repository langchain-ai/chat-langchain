/**
 * Client-side LangSmith wrapper.
 *
 * Calls the MDA LangSmith connector, which performs the LangSmith operation
 * server-side with the workspace key and returns an allowlisted response. The
 * browser sends only its own identity token (Supabase access token or MDA guest
 * token) and never sees LANGSMITH_API_KEY.
 *
 * Route: POST {LANGGRAPH_API_URL}/connectors/langsmith/capabilities/{id}
 */

import { LANGGRAPH_API_URL } from "../constants/api"
import { FEEDBACK_KEY } from "../constants/features"
import { logger } from "../utils/logger"

/** Capability ids declared in `connectors/langsmith.py`. */
const CAPABILITY_FEEDBACK = "langsmith:chat-feedback"
const CAPABILITY_TRACE_VIEWER = "langsmith:trace-viewer"

export interface LangSmithAuth {
  token: string | null | undefined
  region?: string | null
}

interface ApiError {
  error: string
}

function capabilityUrl(capabilityId: string): string {
  const baseUrl = LANGGRAPH_API_URL.replace(/\/$/, "")
  return `${baseUrl}/connectors/langsmith/capabilities/${capabilityId}`
}

function authHeaders(auth: LangSmithAuth): Record<string, string> {
  if (!auth.token) {
    throw new Error(
      "Auth token required for LangSmith connector requests. Ensure auth has loaded first."
    )
  }
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${auth.token}`,
  }
  if (auth.region) {
    headers["X-Supabase-Region"] = auth.region
  }
  const authKey = process.env.NEXT_PUBLIC_LANGGRAPH_AUTH_KEY
  if (authKey) {
    headers["X-Auth-Key"] = authKey
  }
  return headers
}

/**
 * POST a capability action and return its allowlisted `data` payload.
 */
async function callCapability<T>(
  capabilityId: string,
  body: Record<string, unknown>,
  auth: LangSmithAuth
): Promise<T> {
  const response = await fetch(capabilityUrl(capabilityId), {
    method: "POST",
    headers: authHeaders(auth),
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}`
    try {
      const error: ApiError = await response.json()
      errorMessage = error.error || errorMessage
    } catch {
      // Non-JSON error body; keep the status-based message.
    }
    throw new Error(errorMessage)
  }

  const payload = (await response.json()) as { data?: T }
  return payload.data as T
}

/**
 * Create or update thumbs feedback on a LangSmith run.
 */
export async function createOrUpdateFeedback(
  params: {
    runId: string
    score: "positive" | "negative"
    comment?: string
    feedbackId?: string
  },
  auth: LangSmithAuth
): Promise<{ id: string }> {
  const body = params.feedbackId
    ? {
        action: "update",
        feedback_id: params.feedbackId,
        score: params.score,
        comment: params.comment,
      }
    : {
        action: "create",
        run_id: params.runId,
        key: FEEDBACK_KEY,
        score: params.score,
        comment: params.comment,
      }

  return callCapability<{ id: string }>(CAPABILITY_FEEDBACK, body, auth)
}

/**
 * Delete feedback from LangSmith.
 */
export async function deleteFeedback(
  feedbackId: string,
  auth: LangSmithAuth
): Promise<void> {
  await callCapability(
    CAPABILITY_FEEDBACK,
    { action: "delete", feedback_id: feedbackId },
    auth
  )
}

/**
 * Read a redacted run summary from LangSmith.
 */
export async function readRun(runId: string, auth: LangSmithAuth): Promise<any> {
  try {
    return await callCapability<any>(
      CAPABILITY_TRACE_VIEWER,
      { action: "read", run_id: runId },
      auth
    )
  } catch (error: any) {
    logger.error(`[LangSmith] Failed to fetch run ${runId}`, error)
    throw new Error(`Failed to fetch run: ${error.message || "Network error"}`)
  }
}

/**
 * Generate a public trace URL for a LangSmith run.
 */
export async function shareRun(
  runId: string,
  auth: LangSmithAuth
): Promise<string> {
  const data = await callCapability<{ url: string }>(
    CAPABILITY_TRACE_VIEWER,
    { action: "share", run_id: runId },
    auth
  )
  return data.url
}
