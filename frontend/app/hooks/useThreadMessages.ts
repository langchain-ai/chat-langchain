import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Message } from "../types";
import { useLangGraphClient } from "./useLangGraphClient";
import { StreamState, mergeMessagesById } from "./useStreamState";
import { Document } from "@langchain/core/documents";

function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}

function getMessagesWithSources(messages?: Message[], documents?: Document[]) {
  return (messages ?? []).map((message) => ({
    ...message,
    sources: getSources(documents ?? []),
  }));
}

export function getSources(documents: Document[]) {
  const sources = documents.map((doc) => ({
    url: doc.metadata.source,
    title: doc.metadata.title,
  }));
  return sources;
}

export function useThreadMessages(
  threadId: string | null,
  streamState: StreamState | null,
  stopStream?: (clear?: boolean) => void,
) {
  const client = useLangGraphClient();
  const [messages, setMessages] = useState<Message[]>([]);
  const [next, setNext] = useState<string[]>([]);
  const prevStreamStatus = usePrevious(streamState?.status);

  const refreshMessages = useCallback(async () => {
    if (threadId) {
      const { values, next } = await client.threads.getState<{
        messages: Message[];
        documents: Document[];
      }>(threadId);
      const messages = getMessagesWithSources(
        values.messages,
        values.documents,
      );
      setMessages(messages);
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
        const { values, next } = await client.threads.getState<{
          messages: Message[];
          documents: Document[];
        }>(threadId);
        const messages = getMessagesWithSources(
          values.messages,
          values.documents,
        );
        setMessages(messages);
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
      messages: mergeMessagesById(
        messages,
        getMessagesWithSources(streamState?.messages, streamState?.documents),
      ),
      setMessages,
      next,
    }),
    [
      messages,
      streamState?.messages,
      streamState?.documents,
      next,
      refreshMessages,
    ],
  );
}
