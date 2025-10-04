"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  AppendMessage,
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { v4 as uuidv4 } from "uuid";
import { useExternalMessageConverter } from "@assistant-ui/react";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { useToast } from "../hooks/use-toast";
import {
  convertToOpenAIFormat,
  convertLangchainMessages,
} from "../utils/convert_messages";
import { ThreadChat } from "./chat-interface";
import { SelectModel } from "./SelectModel";
import { ThreadHistory } from "./thread-history";
import { Toaster } from "./ui/toaster";
import { useGraphContext } from "../contexts/GraphContext";
import { useQueryState } from "nuqs";

function ChatLangChainComponent(): React.ReactElement {
  const { toast } = useToast();
  const { threadsData, userData, graphData } = useGraphContext();
  const { userId } = userData;
  const { getUserThreads, createThread, getThreadById } = threadsData;
  const { messages, setMessages, streamMessage, switchSelectedThread } =
    graphData;
  const [isRunning, setIsRunning] = useState(false);
  const [threadId, setThreadId] = useQueryState("threadId");

  const hasCheckedThreadIdParam = useRef(false);
  useEffect(() => {
    if (typeof window === "undefined" || hasCheckedThreadIdParam.current)
      return;
    if (!threadId) {
      hasCheckedThreadIdParam.current = true;
      return;
    }

    hasCheckedThreadIdParam.current = true;

    try {
      getThreadById(threadId).then((thread) => {
        if (!thread) {
          setThreadId(null);
          return;
        }

        switchSelectedThread(thread);
      });
    } catch (e) {
      console.error("Failed to fetch thread in query param", e);
      setThreadId(null);
    }
  }, [threadId]);

  const isSubmitDisabled = !userId;

  async function onNew(message: AppendMessage): Promise<void> {
    if (isSubmitDisabled) {
      toast({
        title: "Failed to send message",
        description: "Unable to find user ID. Please try again later.",
      });
      return;
    }
    if (message.content[0]?.type !== "text") {
      throw new Error("Only text messages are supported");
    }

    setIsRunning(true);

    let currentThreadId = threadId;
    if (!currentThreadId) {
      const thread = await createThread(userId);
      if (!thread) {
        toast({
          title: "Error",
          description: "Thread creation failed.",
        });
        return;
      }
      setThreadId(thread.thread_id);
      currentThreadId = thread.thread_id;
    }

    try {
      const humanMessage = new HumanMessage({
        content: message.content[0].text,
        id: uuidv4(),
      });

      setMessages((prevMessages) => [...prevMessages, humanMessage]);

      await streamMessage(currentThreadId, {
        messages: [convertToOpenAIFormat(humanMessage)],
      });
    } finally {
      setIsRunning(false);
      // Re-fetch threads so that the current thread's title is updated.
      await getUserThreads(userId);
    }
  }

  const threadMessages = useExternalMessageConverter<BaseMessage>({
    callback: convertLangchainMessages,
    messages: messages,
    isRunning,
  });

  const runtime = useExternalStoreRuntime({
    messages: threadMessages,
    isRunning,
    onNew,
  });

  return (
    <div className="overflow-hidden w-full flex md:flex-row flex-col relative">
      {messages.length > 0 ? (
        <div className="absolute top-4 right-4 z-10">
          <SelectModel />
        </div>
      ) : null}
      <div>
        <ThreadHistory />
      </div>
      <div className="w-full h-full overflow-hidden">
        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadChat submitDisabled={isSubmitDisabled} messages={messages} />
        </AssistantRuntimeProvider>
      </div>
      <Toaster />
    </div>
  );
}

export const ChatLangChain = React.memo(ChatLangChainComponent);
