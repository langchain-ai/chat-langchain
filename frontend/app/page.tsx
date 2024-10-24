"use client";

import React, { useState } from "react";
import {
  AppendMessage,
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { v4 as uuidv4 } from "uuid";
import { MyThread } from "./components/Primitives";
import { useExternalMessageConverter } from "@assistant-ui/react";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Toaster } from "./components/ui/toaster";
import {
  convertLangchainMessages,
  convertToOpenAIFormat,
} from "./utils/convert_messages";
import { useGraph } from "./hooks/useGraph";
import { ThreadHistory } from "./components/ThreadHistory";
import { useUser } from "./hooks/useUser";
import { useThreads } from "./hooks/useThreads";
import { useToast } from "./hooks/use-toast";
import { SelectModel } from "./components/SelectModel";

export default function ContentComposerChatInterface(): React.ReactElement {
  const { toast } = useToast();
  const { userId } = useUser();
  const {
    userThreads,
    getUserThreads,
    isUserThreadsLoading,
    createThread,
    threadId: currentThread,
    deleteThread,
  } = useThreads(userId);
  const {
    messages,
    setMessages,
    streamMessage,
    switchSelectedThread,
    selectedModel,
    setSelectedModel,
  } = useGraph({
    userId,
    threadId: currentThread,
  });
  const [isRunning, setIsRunning] = useState(false);

  const isSubmitDisabled = !userId || !currentThread;

  async function onNew(message: AppendMessage): Promise<void> {
    if (isSubmitDisabled) {
      let description = "";
      if (!userId) {
        description = "Unable to find user ID. Please try again later.";
      } else if (!currentThread) {
        description =
          "Unable to find current thread ID. Please try again later.";
      }
      toast({
        title: "Failed to send message",
        description,
      });
      return;
    }
    if (message.content[0]?.type !== "text") {
      throw new Error("Only text messages are supported");
    }
    setIsRunning(true);

    try {
      const humanMessage = new HumanMessage({
        content: message.content[0].text,
        id: uuidv4(),
      });

      setMessages((prevMessages) => [...prevMessages, humanMessage]);

      await streamMessage({
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
          <SelectModel
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />
        </div>
      ) : null}
      <div>
        <ThreadHistory
          isUserThreadsLoading={isUserThreadsLoading}
          getUserThreads={getUserThreads}
          isEmpty={messages.length === 0}
          switchSelectedThread={switchSelectedThread}
          currentThread={currentThread}
          userThreads={userThreads}
          userId={userId}
          createThread={createThread}
          deleteThread={(id) => deleteThread(id, () => setMessages([]))}
          clearMessages={() => setMessages([])}
        />
      </div>
      <div className="w-full h-full overflow-hidden">
        <AssistantRuntimeProvider runtime={runtime}>
          <MyThread
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            submitDisabled={isSubmitDisabled}
            messages={messages}
          />
        </AssistantRuntimeProvider>
      </div>
      <Toaster />
    </div>
  );
}
