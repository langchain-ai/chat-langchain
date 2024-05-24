import { useCallback, useEffect, useReducer } from "react";
import { Client } from "@langchain/langgraph-sdk"
// import orderBy from "lodash/orderBy";
// TODO: get rid of it once Thread & other types are exportable by the langgraph SDK
import { Thread } from "../types";

export interface ThreadListProps {
  threads: Thread[] | null;
  createThread: (name: string) => Promise<Thread>;
  updateThread: (
    name: string,
    thread_id: string,
  ) => Promise<Thread>;
  deleteThread: (thread_id: string) => Promise<void>;
}

function threadsReducer(
  state: Thread[] | null,
  action: Thread | Thread[],
): Thread[] | null {
  state = state ?? [];
  if (!Array.isArray(action)) {
    const newThread = action;
    action = [
      ...state.filter((c) => c.thread_id !== newThread.thread_id),
      newThread,
    ];
  }
  return action
  // TODO: figure this one out -- something is breaking with lodash.orderBy
  // return orderBy(action, "updated_at", "desc");
}

export function useThreadList(): ThreadListProps {
  const [threads, setThreads] = useReducer(threadsReducer, null);
  // TODO: move this to a useLanggraphClient hook
  const client = new Client()

  useEffect(() => {
    async function fetchThreads() {
      const threads = await client.threads.search()
      setThreads(threads);
    }

    fetchThreads();
  }, []);

  const createThread = useCallback(async (name: string) => {
    const saved = await client.threads.create({ metadata: { name } });
    setThreads(saved);
    return saved;
  }, []);

  const updateThread = useCallback(
    async (thread_id: string, name: string) => {
      const saved = await client.threads.upsert(thread_id, { metadata: { name }});
      setThreads(saved);
      return saved;
    },
    [],
  );

  const deleteThread = useCallback(
    async (thread_id: string) => {
      await client.threads.delete(thread_id);
      setThreads((threads || []).filter((c: Thread) => c.thread_id !== thread_id));
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