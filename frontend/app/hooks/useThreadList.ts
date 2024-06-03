import { useCallback, useEffect, useReducer, useState } from "react";
import orderBy from "lodash.orderby";
import { Thread } from "@langchain/langgraph-sdk";

import { useLangGraphClient } from "./useLangGraphClient";

const PAGE_SIZE = 50;

export interface ThreadListProps {
  threads: Thread[] | null;
  createThread: (name: string) => Promise<Thread>;
  updateThread: (thread_id: string, name: string) => Promise<Thread>;
  deleteThread: (thread_id: string) => Promise<void>;
  areThreadsLoading: boolean;
  loadMoreThreads: () => void;
}

function threadsReducer(
  state: Thread[] | null,
  action: { type: "add" | "set"; threads: Thread[] },
): Thread[] | null {
  state = state ?? [];
  if (action.type === "set") {
    return orderBy(action.threads, "updated_at", "desc");
  } else {
    const newThreadIds = new Set(
      action.threads.map((thread) => thread.thread_id),
    );
    const existingThreads = state.filter(
      (thread) => !newThreadIds.has(thread.thread_id),
    );
    return orderBy(
      [...existingThreads, ...action.threads],
      "updated_at",
      "desc",
    );
  }
}

export function useThreadList(userId: string): ThreadListProps {
  const [threads, dispatch] = useReducer(threadsReducer, null);
  const [offset, setOffset] = useState(0);
  const [areThreadsLoading, setAreThreadsLoading] = useState(false);
  const client = useLangGraphClient();

  useEffect(() => {
    async function fetchThreads() {
      // wait until the user is set
      if (userId == null) {
        return;
      }

      setAreThreadsLoading(true);
      const fetchedThreads = await client.threads.search({
        offset,
        limit: PAGE_SIZE,
        metadata: {
          userId,
        },
      });
      if (offset === 0) {
        dispatch({ type: "set", threads: fetchedThreads });
      } else {
        dispatch({ type: "add", threads: fetchedThreads });
      }
      setAreThreadsLoading(false);
    }

    fetchThreads();
  }, [offset, userId]);

  const loadMoreThreads = useCallback(() => {
    if (areThreadsLoading) {
      return;
    }
    setOffset((prevOffset) => prevOffset + PAGE_SIZE);
  }, [areThreadsLoading]);

  const createThread = useCallback(async (name: string) => {
    const saved = await client.threads.create({ metadata: { name, userId } });
    dispatch({ type: "add", threads: [saved] });
    return saved;
  }, [userId]);

  const updateThread = useCallback(async (thread_id: string, name: string) => {
    const saved = await client.threads.update(thread_id, {
      metadata: { name, userId },
    });
    dispatch({ type: "add", threads: [saved] });
    return saved;
  }, [userId]);

  const deleteThread = useCallback(
    async (thread_id: string) => {
      await client.threads.delete(thread_id);
      dispatch({
        type: "set",
        threads: (threads || []).filter(
          (c: Thread) => c.thread_id !== thread_id,
        ),
      });
    },
    [threads],
  );

  return {
    threads,
    createThread,
    updateThread,
    deleteThread,
    areThreadsLoading,
    loadMoreThreads,
  };
}
