/**
 * LangGraph Client Factory
 *
 * Creates authenticated LangGraph SDK clients for API requests.
 * All clients include Authorization header with user ID for backend auth.
 */

import { Client } from "@langchain/langgraph-sdk"
import { LANGGRAPH_API_URL, LANGSMITH_API_KEY } from "@/lib/constants/api"

/**
 * Create a LangGraph client instance with authentication.
 *
 * @param userId - User ID for Authorization header (required - backend enforces auth)
 * @throws Error if userId is not provided
 *
 * @example
 * ```typescript
 * const client = createLangGraphClient(userId)
 * const threads = await client.threads.search({ metadata: { user_id: userId } })
 * ```
 */
export function createLangGraphClient(userId: string | undefined): Client {
  if (!userId) {
    throw new Error(
      "User ID required for authentication. Ensure user is logged in before making requests."
    )
  }

  const headers: Record<string, string> = {
    Authorization: `Bearer ${userId}`,
  }

  // Optional public app key for deployments that set LANGGRAPH_AUTH_SECRET.
  const authKey = process.env.NEXT_PUBLIC_LANGGRAPH_AUTH_KEY
  if (authKey) {
    headers["X-Auth-Key"] = authKey
  }

  return new Client({
    apiUrl: LANGGRAPH_API_URL,
    apiKey: LANGSMITH_API_KEY,
    defaultHeaders: headers,
  })
}
