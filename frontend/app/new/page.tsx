"use client";

import React, { useState } from "react";
import {
  AppendMessage,
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { v4 as uuidv4 } from "uuid";
import { MyThread } from "../components/Primitives";
import { useExternalMessageConverter } from "@assistant-ui/react";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Toaster } from "../components/ui/toaster";
import {
  convertLangchainMessages,
  convertToOpenAIFormat,
} from "../utils/convert_messages";
import { useGraph } from "../hooks/useGraph";
import { ThreadHistory } from "../components/ThreadHistory";
import { useUser } from "../hooks/useUser";
import { useThreads } from "../hooks/useThreads";

export default function ContentComposerChatInterface(): React.ReactElement {
  const { userId } = useUser();
  const {
    messages,
    setMessages,
    streamMessage,
    assistantId,
    createThread,
    threadId: currentThread,
    switchSelectedThread,
  } = useGraph(userId);
  const { userThreads } = useThreads(userId);
  const [isRunning, setIsRunning] = useState(false);

  async function onNew(message: AppendMessage): Promise<void> {
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
    <div className="overflow-hidden w-full flex md:flex-row flex-col">
      <div>
        <ThreadHistory
          isEmpty={messages.length === 0}
          switchSelectedThread={switchSelectedThread}
          currentThread={currentThread}
          userThreads={userThreads}
          userId={userId}
          createThread={createThread}
          assistantId={assistantId}
        />
      </div>
      <div className="w-full h-full overflow-hidden">
        <AssistantRuntimeProvider runtime={runtime}>
          <MyThread messages={messages} />
        </AssistantRuntimeProvider>
      </div>
      <Toaster />
    </div>
  );
}
