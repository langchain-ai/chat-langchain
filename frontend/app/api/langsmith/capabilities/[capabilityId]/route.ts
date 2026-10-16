import { Client } from "langsmith/client"
import { LANGGRAPH_API_URL } from "@/lib/constants/api"
import { handleLangSmithCapability } from "@/lib/server/langsmith-capability"

export const runtime = "nodejs"

let langsmithClient: Client | null = null

function getLangSmithClient(): Client | null {
  const apiKey = process.env.LANGSMITH_API_KEY
  if (!apiKey) return null
  if (!langsmithClient) {
    langsmithClient = new Client({
      apiKey,
      ...(process.env.LANGSMITH_ENDPOINT
        ? { apiUrl: process.env.LANGSMITH_ENDPOINT }
        : {}),
    })
  }
  return langsmithClient
}

export async function POST(
  request: Request,
  context: { params: Promise<{ capabilityId: string }> }
) {
  const client = getLangSmithClient()
  if (!client) {
    return Response.json(
      { error: "LangSmith integration is unavailable" },
      { status: 503 }
    )
  }

  const { capabilityId } = await context.params
  return handleLangSmithCapability(request, capabilityId, {
    client,
    fetch,
    langgraphUrl: LANGGRAPH_API_URL,
    langgraphAuthKey:
      process.env.LANGGRAPH_AUTH_SECRET ||
      process.env.NEXT_PUBLIC_LANGGRAPH_AUTH_KEY,
  })
}
