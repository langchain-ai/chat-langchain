import { useCallback, useState } from "react";
import { Config } from "@langchain/langgraph-sdk";
import { Document } from "@langchain/core/documents";

import { Message } from "../types";
import { useLangGraphClient } from "./useLangGraphClient";
import { RESPONSE_FEEDBACK_KEY, SOURCE_CLICK_KEY } from "../utils/constants";

export interface StreamState {
  status: "inflight" | "error" | "done";
  messages?: Message[];
  documents?: Document[];
  feedbackUrls?: Record<string, string[]>;
}

export interface StreamStateProps {
  streamStates: { [threadId: string]: StreamState | null };
  startStream: (
    input: Message[] | null,
    threadId: string,
    assistantId: string,
    config?: Config,
  ) => Promise<void>;
  stopStream?: (threadId: string, clear?: boolean) => void;
}

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

export function useStreamState(): StreamStateProps {
  const [streamStates, setStreamStates] = useState<
    StreamStateProps["streamStates"]
  >({});
  const [controller, setController] = useState<AbortController | null>(null);
  const client = useLangGraphClient();

  const startStream = useCallback(
    async (
      messages: Message[] | null,
      threadId: string,
      assistantId: string,
      config?: Config,
    ) => {
      const controller = new AbortController();
      setController(controller);
      setStreamStates((streamStates) => ({
        ...streamStates,
        [threadId]: { status: "inflight", messages: messages || [] },
      }));

      const stream = client.runs.stream(threadId, assistantId, {
        input: messages == null ? null : { messages },
        config,
        streamMode: ["messages", "values"],
        signal: controller.signal
      });

      for await (const chunk of stream) {
        if (chunk.event === "messages/partial") {
          const chunkMessages = chunk.data as Message[];
          setStreamStates((streamStates) => ({
            ...streamStates,
            [threadId]: {
              ...streamStates[threadId],
              status: "inflight",
              messages: mergeMessagesById(
                streamStates[threadId]?.messages,
                chunkMessages,
              ),
            },
          }));
        } else if (chunk.event === "values") {
          const data = chunk.data as Record<string, any>;
          setStreamStates((streamStates) => ({
            ...streamStates,
            [threadId]: {
              ...streamStates[threadId],
              status: "inflight",
              documents: data["documents"],
              feedbackUrls: data["feedback_urls"]
            },
          }));
        } else if (chunk.event === "error") {
          setStreamStates((streamStates) => ({
            ...streamStates,
            [threadId]: {
              ...streamStates[threadId],
              status: "error",
            },
          }));
        } else if (chunk.event === "end") {
          setStreamStates((streamStates) => ({
            ...streamStates,
            [threadId]: {
              ...streamStates[threadId],
              status: "done",
            },
          }));
        }
      }
    },
    [],
  );

  const stopStream = useCallback(
    (threadId: string, clear: boolean = false) => {
      controller?.abort();
      setController(null);
      if (clear) {
        setStreamStates((streamStates) => ({
          ...streamStates,
          [threadId]: {
            status: "done",
          },
        }));
      } else {
        setStreamStates((streamStates) => ({
          ...streamStates,
          [threadId]: {
            ...streamStates[threadId],
            status: "done",
          },
        }));
      }
    },
    [controller],
  );

  return {
    startStream,
    stopStream,
    streamStates,
  };
}
