import { useEffect, useState } from "react";

import { Client, Thread } from "@langchain/langgraph-sdk";

export const createClient = () => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000/api";
  return new Client({
    apiUrl,
  });
};

export interface ThreadActual extends Thread {
  values: Record<string, any> | undefined;
  config: Record<string, any>;
  status: string;
}

export function useThreads(userId: string | undefined) {
  const [isUserThreadsLoading, setIsUserThreadsLoading] = useState(false);
  const [userThreads, setUserThreads] = useState<ThreadActual[]>([]);

  useEffect(() => {
    if (typeof window == "undefined" || !userId) return;
    getUserThreads(userId);
  }, [userId]);

  const getUserThreads = async (id: string) => {
    setIsUserThreadsLoading(true);
    try {
      const client = createClient();

      const userThreads = (await client.threads.search({
        metadata: {
          user_id: id,
        },
        limit: 100,
      })) as Awaited<ThreadActual[]>;

      if (userThreads.length > 0) {
        const lastInArray = userThreads[0];
        const allButLast = userThreads.slice(1, userThreads.length);
        const filteredThreads = allButLast.filter(
          (thread) => thread.values && Object.keys(thread.values).length > 0,
        );
        setUserThreads([...filteredThreads, lastInArray]);
      }
    } finally {
      setIsUserThreadsLoading(false);
    }
  };

  const getThreadById = async (id: string) => {
    const client = createClient();
    return (await client.threads.get(id)) as Awaited<ThreadActual>;
  };

  return {
    isUserThreadsLoading,
    userThreads,
    getThreadById,
    getUserThreads,
  };
}
