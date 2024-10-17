"use client";

import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useMessage,
  useThreadRuntime,
} from "@assistant-ui/react";
import { type FC } from "react";
import NextImage from "next/image";

import { ArrowDownIcon, SendHorizontalIcon } from "lucide-react";
import { MarkdownText } from "./ui/assistant-ui/markdown-text";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";
import { BaseMessage } from "@langchain/core/messages";
import { cn } from "../utils/cn";
import { useGeneratingQuestionsUI } from "./GeneratingQuestionsToolUI";
import { useAnswerHeaderToolUI } from "./AnswerHeaderToolUI";
import { useProgressToolUI } from "./ProgressToolUI";
import { useSelectedDocumentsUI } from "./SelectedDocumentsToolUI";
import { useRouterLogicUI } from "./RouterLogicToolUI";
import { SuggestedQuestions } from "./SuggestedQuestions";
import { ModelOptions } from "../types";
import { SelectModel } from "./SelectModel";

export interface MyThreadProps extends MyComposerProps {
  selectedModel: ModelOptions;
  setSelectedModel: (model: ModelOptions) => void;
}

export const MyThread: FC<MyThreadProps> = (props: MyThreadProps) => {
  const isEmpty = props.messages.length === 0;

  useGeneratingQuestionsUI();
  useAnswerHeaderToolUI();
  useProgressToolUI();
  useSelectedDocumentsUI();
  useRouterLogicUI();

  return (
    <ThreadPrimitive.Root className="flex flex-col h-screen overflow-hidden w-full">
      {!isEmpty ? (
        <ThreadPrimitive.Viewport
          className={cn(
            "flex-1 overflow-y-auto scroll-smooth bg-inherit transition-all duration-300 ease-in-out w-full",
            isEmpty ? "pb-[30vh] sm:pb-[50vh]" : "pb-32 sm:pb-24",
            "scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent",
          )}
        >
          <div className="md:pl-8 lg:pl-24 mt-2 max-w-full">
            <ThreadPrimitive.Messages
              components={{
                UserMessage: MyUserMessage,
                AssistantMessage: MyAssistantMessage,
              }}
            />
          </div>
        </ThreadPrimitive.Viewport>
      ) : null}
      <MyThreadScrollToBottom />
      {isEmpty ? (
        <div className="flex items-center justify-center flex-grow my-auto">
          <div className="flex flex-col items-center mx-4 md:mt-0 mt-24">
            <div className="flex flex-row gap-1 items-center justify-center">
              <p className="text-xl sm:text-2xl">Chat LangChain</p>
              <NextImage
                src="/images/lc_logo.jpg"
                className="rounded-3xl"
                alt="LangChain Logo"
                width={32}
                height={32}
                style={{ width: "auto", height: "auto" }}
              />
            </div>
            <div className="mb-4 sm:mb-[24px] mt-1 sm:mt-2">
              <SelectModel
                selectedModel={props.selectedModel}
                setSelectedModel={props.setSelectedModel}
              />
            </div>
            <div className="md:mb-8 mb-4">
              <SuggestedQuestions />
            </div>
            <MyComposer
              submitDisabled={props.submitDisabled}
              messages={props.messages}
            />
          </div>
        </div>
      ) : (
        <MyComposer
          submitDisabled={props.submitDisabled}
          messages={props.messages}
        />
      )}
    </ThreadPrimitive.Root>
  );
};

const MyThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="absolute bottom-28 left-1/2 transform -translate-x-1/2 rounded-full disabled:invisible bg-white bg-opacity-75"
      >
        <ArrowDownIcon className="text-gray-600 hover:text-gray-800 transition-colors ease-in-out" />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

interface MyComposerProps {
  messages: BaseMessage[];
  submitDisabled: boolean;
}

const MyComposer: FC<MyComposerProps> = (props: MyComposerProps) => {
  const isEmpty = props.messages.length === 0;

  return (
    <ComposerPrimitive.Root
      className={cn(
        "focus-within:border-aui-ring/20 flex w-full items-center md:justify-left justify-center rounded-lg border px-2.5 py-2.5 shadow-sm transition-all duration-300 ease-in-out bg-[#282828] border-gray-600",
        isEmpty ? "" : "md:ml-24 ml-3 mb-6",
        isEmpty ? "w-full" : "md:w-[70%] w-[95%] md:max-w-[832px]",
      )}
    >
      <ComposerPrimitive.Input
        autoFocus
        placeholder="How can I..."
        rows={1}
        className="placeholder:text-gray-400 text-gray-100 max-h-40 flex-1 resize-none border-none bg-transparent px-2 py-2 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />
      <div className="flex-shrink-0">
        <ThreadPrimitive.If running={false} disabled={props.submitDisabled}>
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
    <MessagePrimitive.Root className="pt-2 sm:pt-4 flex w-full md:max-w-4xl md:mx-0 mx-auto max-w-[95%] md:py-4 py-2">
      <div className="bg-inherit text-white break-words rounded-2xl sm:rounded-3xl pt-2 md:pt-2.5 mb-[-15px] sm:mb-[-25px] text-2xl sm:text-4xl font-light">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

const MyAssistantMessage: FC = () => {
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
