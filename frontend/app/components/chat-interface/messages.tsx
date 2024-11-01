"use client";

import {
  MessagePrimitive,
  useMessage,
  useThreadRuntime,
} from "@assistant-ui/react";
import { type FC } from "react";

import { MarkdownText } from "../ui/assistant-ui/markdown-text";

export const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="pt-2 sm:pt-4 flex w-full md:max-w-4xl md:mx-0 mx-auto max-w-[95%] md:py-4 py-2">
      <div className="bg-inherit text-white break-words rounded-2xl sm:rounded-3xl pt-2 md:pt-2.5 mb-[-15px] sm:mb-[-25px] text-2xl sm:text-4xl font-light">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

export const AssistantMessage: FC = () => {
  const threadRuntime = useThreadRuntime();
  const threadState = threadRuntime.getState();
  const isLast = useMessage((m) => m.isLast);
  const shouldRenderMessageBreak =
    threadState.messages.filter((msg) => msg.role === "user")?.length > 1 &&
    !isLast;

  return (
    <MessagePrimitive.Root className="flex w-full md:max-w-4xl md:mx-0 mx-auto max-w-[95%] md:py-4 py-2">
      <div className="bg-inherit text-white max-w-full sm:max-w-3xl break-words leading-6 sm:leading-7">
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
        {shouldRenderMessageBreak ? (
          <hr className="relative left-1/2 -translate-x-1/2 w-[90vw] sm:w-[45vw] mt-4 sm:mt-6 border-gray-600" />
        ) : null}
      </div>
    </MessagePrimitive.Root>
  );
};
