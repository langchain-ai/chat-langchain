/**
 * Client-side LangSmith API wrapper
 * Proxies requests through LangGraph Server to keep API keys server-side
 */

import { LANGGRAPH_API_URL, LANGSMITH_API_KEY } from "../constants/api"
import { logger } from "../utils/logger"

interface ApiError {
  error: string
}

/**
 * Get base URL for LangSmith API routes
 * Uses LangGraph Server URL since routes are mounted there
 */
function getLangSmithApiUrl(): string {
  // Remove trailing slash if present, then append /langsmith
  const baseUrl = LANGGRAPH_API_URL.replace(/\/$/, "")
  const url = `${baseUrl}/langsmith`

  logger.debug('[LangSmith] API URL:', url)

  return url
}

/**
 * Get auth headers for LangGraph Server requests
 * Includes API key if available (for LangGraph Server auth)
 */
function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  }

  // Add API key if available (for LangGraph Server authentication)
  // Uses same key as LangGraph client for consistency
  if (LANGSMITH_API_KEY) {
    headers["x-api-key"] = LANGSMITH_API_KEY
  }

  return headers
}

/**
 * Handle API response errors consistently
 */
async function handleApiResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}`
    try {
      const error: ApiError = await response.json()
      errorMessage = error.error || errorMessage
    } catch {
      // If JSON parsing fails, try to get text
      try {
        const text = await response.text()
        errorMessage = text || errorMessage
      } catch {
        // Fallback to status code
      }
    }
    throw new Error(errorMessage)
  }
  return response.json()
}

/**
 * Create or update feedback on a LangSmith run
 */
export async function createOrUpdateFeedback(params: {
  runId: string
  feedbackKey: string
  score: "positive" | "negative"
  comment?: string
  feedbackId?: string
}): Promise<{ id: string }> {
  const response = await fetch(`${getLangSmithApiUrl()}/feedback`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({
      runId: params.runId,
      feedbackKey: params.feedbackKey,
      score: params.score,
      comment: params.comment,
      feedbackId: params.feedbackId,
    }),
  })

  return handleApiResponse(response)
}

/**
 * Delete feedback from LangSmith
 */
export async function deleteFeedback(feedbackId: string): Promise<void> {
  const response = await fetch(
    `${getLangSmithApiUrl()}/feedback?feedbackId=${encodeURIComponent(feedbackId)}`,
    {
      method: "DELETE",
      headers: getAuthHeaders(),
    }
  )

  await handleApiResponse(response)
}

/**
 * Read run details from LangSmith
 */
export async function readRun(runId: string): Promise<any> {
  const url = `${getLangSmithApiUrl()}/runs/${encodeURIComponent(runId)}`
  
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: getAuthHeaders(),
    })

    return handleApiResponse(response)
  } catch (error: any) {
    logger.error(`[LangSmith] Failed to fetch run ${runId}`, error)
    throw new Error(`Failed to fetch run: ${error.message || "Network error"}`)
  }
}

/**
 * Generate a public trace URL for a LangSmith run.
 */
export async function shareRun(runId: string): Promise<string> {
  const response = await fetch(
    `${getLangSmithApiUrl()}/runs/${encodeURIComponent(runId)}/share`,
    {
      method: "POST",
      headers: getAuthHeaders(),
    }
  )

  const data = await handleApiResponse<{ shareUrl: string }>(response)
  return data.shareUrl
}

