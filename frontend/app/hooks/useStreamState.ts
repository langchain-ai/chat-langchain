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
  feedbackUrls?: Record<string, string>;
}

export interface StreamStateProps {
  streamState: StreamState | null;
  startStream: (
    input: Message[] | null,
    threadId: string,
    assistantId: string,
    config?: Config,
  ) => Promise<void>;
  stopStream?: (clear?: boolean) => void;
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
  const [current, setCurrent] = useState<StreamState | null>(null);
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
      setCurrent({ status: "inflight", messages: messages || [] });

      const stream = client.runs.stream(threadId, assistantId, {
        input: messages == null ? null : { messages },
        config,
        streamMode: ["messages", "values"],
        signal: controller.signal,
        feedbackKeys: [RESPONSE_FEEDBACK_KEY, SOURCE_CLICK_KEY],
      });

      for await (const chunk of stream) {
        if (chunk.event === "messages/partial") {
          const chunkMessages = chunk.data as Message[];
          setCurrent((current) => ({
            ...current,
            status: "inflight",
            messages: mergeMessagesById(current?.messages, chunkMessages),
          }));
        } else if (chunk.event === "values") {
          const data = chunk.data as Record<string, any>;
          setCurrent((current) => ({
            ...current,
            status: "inflight",
            documents: data["documents"],
          }));
        } else if (chunk.event === "error") {
          setCurrent((current) => ({
            ...current,
            status: "error",
          }));
        } else if (chunk.event === "feedback") {
          setCurrent((current) => ({
            ...current,
            feedbackUrls: chunk.data,
            status: "inflight",
          }));
        } else if (chunk.event === "end") {
          setCurrent((current) => ({
            ...current,
            status: "done",
          }));
        }
      }
    },
    [],
  );

  const stopStream = useCallback(
    (clear: boolean = false) => {
      controller?.abort();
      setController(null);
      if (clear) {
        setCurrent((current) => ({
          ...current,
          status: "done",
        }));
      } else {
        setCurrent((current) => ({
          ...current,
          status: "done",
          messages: current?.messages,
        }));
      }
    },
    [controller],
  );

  return {
    startStream,
    stopStream,
    streamState: current,
  };
}
