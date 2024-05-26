import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Message } from "../types";
import { useLangGraphClient } from "./useLangGraphClient";
import { StreamState, getSources, mergeMessagesById } from "./useStreamState";

function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}

function getMessagesFromValues (values: Record<string, any>) {
  const messages = ((Array.isArray(values) ? values : values.messages) ?? []) as Message[];
  const sources = getSources(values)
  return messages.map(message => ({ ...message, sources }))
}

export function useThreadMessages(
  threadId: string | null,
  streamState: StreamState | null,
  stopStream?: (clear?: boolean) => void,
) {
  const client = useLangGraphClient()
  const [messages, setMessages] = useState<Message[]>([]);
  const [next, setNext] = useState<string[]>([]);
  const prevStreamStatus = usePrevious(streamState?.status);

  const refreshMessages = useCallback(async () => {
    if (threadId) {
      const { values, next } = await client.threads.getState(threadId);
      const messages = getMessagesFromValues(values);
      setMessages(messages)
      setNext(next);
    }
  }, [threadId]);

  useEffect(() => {
    refreshMessages();
    return () => {
      setMessages([]);
    };
  }, [threadId, refreshMessages]);

  useEffect(() => {
    async function fetchMessages() {
      if (threadId) {
        const { values, next } = await client.threads.getState(threadId);
        const messages = getMessagesFromValues(values);
        setMessages(messages)
        setNext(next);
        stopStream?.(true);
      }
    }

    if (prevStreamStatus === "inflight" && streamState?.status !== "inflight") {
      setNext([]);
      fetchMessages();
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamState?.status]);

  return useMemo(
    () => ({
      refreshMessages,
      messages: mergeMessagesById(messages, streamState?.messages),
      setMessages,
      next,
    }),
    [messages, streamState?.messages, next, refreshMessages],
  );
}