import { useCallback, useState } from "react";
import { Message, Source } from "../types";
import { useLangGraphClient } from "./useLangGraphClient";
import { Config } from "@langchain/langgraph-sdk";
import { Document } from "@langchain/core/documents";

export interface StreamState {
  status: "inflight" | "error" | "done";
  messages?: Message[] | Record<string, any>;
  run_id?: string;
}

export interface StreamStateProps {
  streamState: StreamState | null;
  startStream: (
    input: Message[],
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

export function getSources(values: Record<string, any>) {
  const documents = (values["documents"] ?? []) as Array<Document>;
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

  // TODO: figure out how we actually deal with the controller here
  const startStream = useCallback(
    async (
      messages: Message[],
      threadId: string,
      assistantId: string,
      config?: Config,
    ) => {
      const controller = new AbortController();
      setController(controller);
      setCurrent({ status: "inflight", messages: messages || [] });

      const stream = client.runs.stream(threadId, assistantId, {
        input: { messages },
        config,
        streamMode: ["messages", "values"],
      });

      let sources: Source[] = [];
      for await (const chunk of stream) {
        if (chunk.event === "metadata") {
          setCurrent((current) => ({
            status: "inflight",
            messages: current?.messages,
            run_id: chunk.data["run_id"] as string,
          }));
        } else if (chunk.event === "messages/partial") {
          const chunkMessages = chunk.data as Message[];
          setCurrent((current) => ({
            status: "inflight",
            messages: mergeMessagesById(
              current?.messages,
              chunkMessages.map((message) => ({ ...message, sources })),
            ),
            run_id: current?.run_id,
          }));
        } else if (chunk.event === "values") {
          const data = chunk.data as Record<string, any>;
          sources = getSources(data);
        } else if (chunk.event === "error") {
          setCurrent((current) => ({
            status: "error",
            messages: current?.messages,
            run_id: current?.run_id,
          }));
        } else if (chunk.event === "end") {
          setCurrent((current) => ({
            status: "done",
            messages: current?.messages,
            run_id: current?.run_id,
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
