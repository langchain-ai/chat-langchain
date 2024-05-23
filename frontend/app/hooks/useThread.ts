import { useQuery, useQueryClient } from "react-query";
import { useSearchParams } from "next/navigation"
import { Client } from "@langchain/langgraph-sdk"

export function useThread() {
  // Extract route parameters
  const searchParams = useSearchParams();
  const threadId = searchParams.get("thread_id");
  const queryClient = useQueryClient();
  const langgraphClient = new Client();

  // React Query to fetch chat details if chatId is present
  const { data: currentChat, isLoading } = useQuery(
    ["thread", threadId],
    async () => await langgraphClient.threads.get(threadId as string),
    {
      enabled: !!threadId,
    },
  );

  const invalidateChat = (threadId: string) => {
    queryClient.invalidateQueries(["thread", threadId]);
  };

  // Return both loading states, the chat data, and the assistant configuration
  return {
    currentChat,
    isLoading,
    invalidateChat,
  };
}