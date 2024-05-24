import { useCallback, useEffect, useMemo, useState } from "react";
import { Message, Source } from "../types";
import { Document } from "@langchain/core/documents";
import { Client } from "@langchain/langgraph-sdk";

export function mergeMessagesById(
  left: Message[] | Record<string, any> | null | undefined,
  right: Message[] | Record<string, any> | null | undefined,
): Message[] {
  const leftMsgs = Array.isArray(left) ? left : left?.messages;
  const rightMsgs = Array.isArray(right) ? right : right?.messages;

  const merged = (leftMsgs ?? [])?.slice();
  for (const msg of rightMsgs ?? []) {
    const foundIdx = merged.findIndex((m: any) => m.id === msg.id);
    if (foundIdx === -1) {
      merged.push(msg);
    } else {
      merged[foundIdx] = msg;
    }
  }
  return merged;
}

function getSources(values: Record<string, any>) {
  const documents = (values["documents"] ?? []) as Array<Document>;
  const sources = documents.map((doc) => ({
    url: doc.metadata.source,
    title: doc.metadata.title,
  }));
  return sources
}

export function useThreadMessages(
  threadId: string | null,
) {
  // TODO: move this into useLanggraphClient hook
  const client = new Client()
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    async function fetchMessages() {
      if (threadId) {
        const { values } = await client.threads.getState(threadId);
        if (values != null) {
          const messages = (Array.isArray(values) ? values : values.messages ?? []) as Message[];
          const sources = getSources(values)
          setMessages(messages.map(message => ( {...message, sources })));
        }
      }
    } 
    fetchMessages()
  }, [threadId])

  const updateMessages = useCallback(async (
    stream: AsyncGenerator<Record<string, any>>
  ) => {
    let sources: Source[] = []
    for await (const chunk of stream) {
      if (chunk.event === "messages/partial") {
        const chunkMessages = chunk.data as Message[];
        setMessages((prevMessages) =>
          mergeMessagesById(
            prevMessages,
            chunkMessages.map((message) => ({ ...message, sources })),
          ),
        );
      } else if (chunk.event === "values") {
        const data = chunk.data as Record<string, any>;
        sources = getSources(data);
      }
    }
  }, [threadId])

  return useMemo(
    () => ({
      updateMessages,
      messages,
      setMessages
    }),
    [messages, updateMessages],
  );
}