/**
 * LangGraph Client Factory
 *
 * Creates authenticated LangGraph SDK clients for API requests.
 * All clients include Authorization header with a verifiable backend credential.
 */

import { Client } from "@langchain/langgraph-sdk"
import { LANGGRAPH_API_URL, LANGSMITH_API_KEY } from "@/lib/constants/api"
import type { AuthRegion } from "@/lib/auth"

/**
 * Create a LangGraph client instance with authentication.
 *
 * @param authToken - Supabase access token, guest token, or legacy user ID.
 * @param authRegion - Supabase region used for signed-in token verification.
 * @throws Error if authToken is not provided
 *
 * @example
 * ```typescript
 * const client = createLangGraphClient(authToken)
 * const threads = await client.threads.search({ metadata: { user_id: userId } })
 * ```
 */
export function createLangGraphClient(
  authToken: string | undefined,
  authRegion?: AuthRegion
): Client {
  if (!authToken) {
    throw new Error(
      "Auth token required for LangGraph requests. Ensure auth has loaded before making requests."
    )
  }

  const headers: Record<string, string> = {
    Authorization: `Bearer ${authToken}`,
  }

  // Optional public app key for deployments that set LANGGRAPH_AUTH_SECRET.
  const authKey = process.env.NEXT_PUBLIC_LANGGRAPH_AUTH_KEY
  if (authKey) {
    headers["X-Auth-Key"] = authKey
  }
  if (authRegion) {
    headers["X-Supabase-Region"] = authRegion
  }

  return new Client({
    apiUrl: LANGGRAPH_API_URL,
    apiKey: LANGSMITH_API_KEY,
    defaultHeaders: headers,
  })
}
