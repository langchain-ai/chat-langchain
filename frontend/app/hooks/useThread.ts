import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "react-query";
import { useSearchParams } from "next/navigation";

import { useLangGraphClient } from "./useLangGraphClient";

export function useThread(userId: string) {
  // Extract route parameters
  const [threadId, setThreadId] = useState<string>();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const langGraphClient = useLangGraphClient();

  const previousThreadId = useRef<string>();
  useEffect(() => {
    const threadId = searchParams.get("threadId") as string;
    if (threadId !== previousThreadId.current) {
      setThreadId(threadId);
    }
    previousThreadId.current = threadId;
  }, [searchParams]);

  const getThread = async () => {
    const thread = await langGraphClient.threads.get(threadId as string);
    if (thread?.metadata?.["userId"] !== userId) {
      return null;
    }
    return thread;
  };

  // React Query to fetch chat details if chatId is present
  const { data: currentThread, isLoading } = useQuery(
    ["thread", threadId],
    getThread,
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
