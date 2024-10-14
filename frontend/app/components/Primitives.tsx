"use client";

import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useComposerStore,
  useMessageStore,
  useThreadRuntime,
} from "@assistant-ui/react";
import { type FC } from "react";

import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar";
import { Button } from "./ui/button";
import { ArrowDownIcon, SendHorizontalIcon } from "lucide-react";
import { MarkdownText } from "./ui/assistant-ui/markdown-text";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";
import { BaseMessage } from "@langchain/core/messages";
import { cn } from "../utils/cn";
import { useGeneratingQuestionsUI } from "./GeneratingQuestionsToolUI";
import { useAnswerHeaderToolUI } from "./AnswerHeaderToolUI";

export interface MyThreadProps extends MyComposerProps {}

export const MyThread: FC<MyThreadProps> = (props: MyThreadProps) => {
  const isEmpty = props.messages.length === 0;

  useGeneratingQuestionsUI();
  useAnswerHeaderToolUI();

  return (
    <ThreadPrimitive.Root className="flex flex-col h-full relative">
      <ThreadPrimitive.Viewport
        className={cn(
          "flex-1 overflow-y-auto scroll-smooth bg-inherit px-4 pt-8 transition-all duration-300 ease-in-out w-full",
          isEmpty ? "pb-[50vh]" : "pb-20",
        )}
      >
        <ThreadPrimitive.Messages
          components={{
            UserMessage: MyUserMessage,
            EditComposer: MyEditComposer,
            AssistantMessage: MyAssistantMessage,
          }}
        />
      </ThreadPrimitive.Viewport>
      <MyThreadScrollToBottom />
      <MyComposer messages={props.messages} />
    </ThreadPrimitive.Root>
  );
};

const MyThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="absolute -top-8 rounded-full disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

interface MyComposerProps {
  messages: BaseMessage[];
}

const MyComposer: FC<MyComposerProps> = (props: MyComposerProps) => {
  const isEmpty = props.messages.length === 0;

  return (
    <ComposerPrimitive.Root
      className={cn(
        "focus-within:border-aui-ring/20 flex w-full items-center rounded-lg border px-2.5 py-2.5 shadow-sm transition-all duration-300 ease-in-out bg-white",
        "absolute left-1/2 transform -translate-x-1/2 max-w-2xl",
        isEmpty ? "top-1/2 -translate-y-1/2" : "bottom-4",
      )}
    >
      <ComposerPrimitive.Input
        autoFocus
        placeholder="Write a message..."
        rows={1}
        className="placeholder:text-muted-foreground max-h-40 flex-1 resize-none border-none bg-transparent px-2 py-2 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />
      <div className="flex-shrink-0">
        <ThreadPrimitive.If running={false}>
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              tooltip="Send"
              variant="default"
              className="my-1 size-8 p-2 transition-opacity ease-in"
            >
              <SendHorizontalIcon />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel asChild>
            <TooltipIconButton
              tooltip="Cancel"
              variant="default"
              className="my-1 size-8 p-2 transition-opacity ease-in"
            >
              <CircleStopIcon />
            </TooltipIconButton>
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
      </div>
    </ComposerPrimitive.Root>
  );
};

const MyUserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="w-full max-w-2xl py-4 mx-auto">
      <div className="bg-inherit text-gray-200 max-w-xl break-words rounded-3xl px-5 py-2.5 text-4xl font-light">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

const MyComposerSend = () => {
  const messageStore = useMessageStore();
  const composerStore = useComposerStore();
  const threadRuntime = useThreadRuntime();

  const handleSend = () => {
    const composerState = composerStore.getState();
    const { parentId, message } = messageStore.getState();

    threadRuntime.append({
      parentId,
      role: message.role,
      content: [{ type: "text", text: composerState.text }],
    });
    composerState.cancel();
  };

  return <Button onClick={handleSend}>Save</Button>;
};

const MyEditComposer: FC = () => {
  return (
    <ComposerPrimitive.Root className="bg-muted my-4 flex w-full max-w-2xl flex-col gap-2 rounded-xl">
      <ComposerPrimitive.Input
        className="text-foreground flex h-8 w-full resize-none border-none bg-transparent p-4 pb-0 outline-none focus:ring-0"
        // Don't submit on `Enter`, instead add a newline.
        submitOnEnter={false}
      />

      <div className="mx-3 mb-3 flex items-center justify-center gap-2 self-end">
        <ComposerPrimitive.Cancel asChild>
          <Button variant="ghost">Cancel</Button>
        </ComposerPrimitive.Cancel>
        <MyComposerSend />
      </div>
    </ComposerPrimitive.Root>
  );
};

const MyAssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="relative flex w-full max-w-2xl py-4 mx-auto">
      <div className="ml-6 bg-inherit text-white max-w-xl break-words leading-7">
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
      </div>
    </MessagePrimitive.Root>
  );
};

const CircleStopIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      width="16"
      height="16"
    >
      <rect width="10" height="10" x="3" y="3" rx="2" />
    </svg>
  );
};
