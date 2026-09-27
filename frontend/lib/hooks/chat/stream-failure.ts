import type { Client } from "@langchain/langgraph-sdk"

export const STREAM_FAILURE_MESSAGE =
  "Sorry, I couldn't complete that request. Please try again."

export async function persistStreamFailure(
  client: Client,
  threadId: string,
  assistantMessageId: string,
): Promise<void> {
  await client.threads.updateState(threadId, {
    values: {
      messages: [
        {
          id: assistantMessageId,
          role: "assistant",
          content: STREAM_FAILURE_MESSAGE,
        },
      ],
    },
  })
}
