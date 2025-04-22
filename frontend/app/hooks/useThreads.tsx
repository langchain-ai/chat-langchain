"use client";

import { useEffect, useState } from "react";

import { Client, Thread } from "@langchain/langgraph-sdk";
import { useToast } from "./use-toast";
import { useQueryState } from "nuqs";

export const createClient = () => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000/api";
  return new Client({
    apiUrl,
  });
};

export function useThreads(userId: string | undefined) {
  const { toast } = useToast();
  const [isUserThreadsLoading, setIsUserThreadsLoading] = useState(false);
  const [userThreads, setUserThreads] = useState<Thread[]>([]);
  const [threadId, setThreadId] = useQueryState("threadId");

  useEffect(() => {
    if (typeof window == "undefined" || !userId) return;
    getUserThreads(userId);
  }, [userId]);

  const createThread = async (id: string) => {
    const client = createClient();
    let thread;
    try {
      thread = await client.threads.create({
        metadata: {
          user_id: id,
        },
      });
      if (!thread || !thread.thread_id) {
        throw new Error("Thread creation failed.");
      }
      setThreadId(thread.thread_id);
    } catch (e) {
      console.error("Error creating thread", e);
      toast({
        title: "Error creating thread.",
      });
    }
    return thread;
  };

  const getUserThreads = async (id: string) => {
    setIsUserThreadsLoading(true);
    try {
      const client = createClient();

      const userThreads = (await client.threads.search({
        metadata: {
          user_id: id,
        },
        limit: 100,
      })) as Awaited<Thread[]>;

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
    return (await client.threads.get(id)) as Awaited<Thread>;
  };

  const deleteThread = async (id: string, clearMessages: () => void) => {
    if (!userId) {
      throw new Error("User ID not found");
    }
    setUserThreads((prevThreads) => {
      const newThreads = prevThreads.filter(
        (thread) => thread.thread_id !== id,
      );
      return newThreads;
    });
    const client = createClient();
    await client.threads.delete(id);
    if (id === threadId) {
      // Remove the threadID from query params, and refetch threads to
      // update the sidebar UI.
      clearMessages();
      getUserThreads(userId);
      setThreadId(null);
    }
  };

  return {
    isUserThreadsLoading,
    userThreads,
    getThreadById,
    setUserThreads,
    getUserThreads,
    createThread,
    deleteThread,
  };
}
