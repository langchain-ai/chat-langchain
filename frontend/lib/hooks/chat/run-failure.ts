import type { Client } from "@langchain/langgraph-sdk"

export const RUN_FAILURE_MESSAGE =
  "The run failed or reached its step limit, so your question was not answered. Please try again."

export async function appendRunFailureMessage(
  client: Pick<Client, "threads">,
  threadId: string,
  messageId: string,
  content: string
): Promise<void> {
  await client.threads.updateState(threadId, {
    values: {
      messages: [{
        id: messageId,
        type: "ai",
        role: "assistant",
        content,
      }],
    },
  } as any)
}
