import { useCallback, useState } from "react";
import { Message } from "../types";
import { useLangGraphClient } from "./useLangGraphClient";
import { Config } from "@langchain/langgraph-sdk";
import { Document } from "@langchain/core/documents";

export interface StreamState {
  status: "inflight" | "error" | "done";
  messages?: Message[];
  documents?: Document[];
  run_id?: string;
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

export function getSources(documents: Document[]) {
  const sources = documents.map((doc) => ({
    url: doc.metadata.source,
    title: doc.metadata.title,
  }));
  return sources;
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
      });

      for await (const chunk of stream) {
        if (chunk.event === "metadata") {
          setCurrent((current) => ({
            ...current,
            status: "inflight",
            run_id: chunk.data["run_id"] as string,
          }));
        } else if (chunk.event === "messages/partial") {
          const chunkMessages = chunk.data as Message[];
          const sources = getSources(current?.documents ?? []);
          setCurrent((current) => ({
            ...current,
            status: "inflight",
            messages: mergeMessagesById(
              current?.messages,
              chunkMessages.map((message) => ({ ...message, sources })),
            ),
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
          status: "done",
          run_id: current?.run_id,
        }));
      } else {
        setCurrent((current) => ({
          status: "done",
          messages: current?.messages,
          run_id: current?.run_id,
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
