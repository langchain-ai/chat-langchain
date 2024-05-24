import { useQuery, useQueryClient } from "react-query";
import { useSearchParams } from "next/navigation"
import { Client } from "@langchain/langgraph-sdk"
import { useEffect, useState } from "react";

export function useThread() {
  // Extract route parameters
  const [threadId, setThreadId] = useState<string>()
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const langgraphClient = new Client();

  useEffect(() => {
    setThreadId(searchParams.get("threadId") as string)
  }, [searchParams])

  // React Query to fetch chat details if chatId is present
  const { data: currentThread, isLoading } = useQuery(
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
    currentThread,
    isLoading,
    invalidateChat,
  };
}