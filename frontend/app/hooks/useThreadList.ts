import { useCallback, useEffect, useReducer } from "react";
import orderBy from "lodash.orderby";
import { Thread } from "@langchain/langgraph-sdk";

import { useLangGraphClient } from "./useLangGraphClient";

export interface ThreadListProps {
  threads: Thread[] | null;
  createThread: (name: string) => Promise<Thread>;
  updateThread: (thread_id: string, name: string) => Promise<Thread>;
  deleteThread: (thread_id: string) => Promise<void>;
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

export function useThreadList(): ThreadListProps {
  const [threads, dispatch] = useReducer(threadsReducer, null);
  const client = useLangGraphClient();

  useEffect(() => {
    async function fetchThreads(offset = 0, limit = 20) {
      const fetchedThreads = await client.threads.search({ offset, limit });
      if (offset === 0) {
        dispatch({ type: "set", threads: fetchedThreads });
      } else {
        dispatch({ type: "add", threads: fetchedThreads });
      }
    }

    fetchThreads();
  }, []);

  const createThread = useCallback(async (name: string) => {
    const saved = await client.threads.create({ metadata: { name } });
    dispatch({ type: "add", threads: [saved] });
    return saved;
  }, []);

  const updateThread = useCallback(async (thread_id: string, name: string) => {
    const saved = await client.threads.upsert(thread_id, {
      metadata: { name },
    });
    dispatch({ type: "add", threads: [saved] });
    return saved;
  }, []);

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
  };
}
