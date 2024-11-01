"use client";

import { ThreadPrimitive } from "@assistant-ui/react";
import { type FC } from "react";
import NextImage from "next/image";

import { ArrowDownIcon } from "lucide-react";
import { useAnswerHeaderToolUI } from "../AnswerHeaderToolUI";
import { useGeneratingQuestionsUI } from "../GeneratingQuestionsToolUI";
import { useProgressToolUI } from "../ProgressToolUI";
import { useRouterLogicUI } from "../RouterLogicToolUI";
import { useSelectedDocumentsUI } from "../SelectedDocumentsToolUI";
import { SelectModel } from "../SelectModel";
import { SuggestedQuestions } from "../SuggestedQuestions";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { AssistantMessage, UserMessage } from "./messages";
import { ChatComposer, ChatComposerProps } from "./chat-composer";
import { cn } from "@/app/utils/cn";

export interface ThreadChatProps extends ChatComposerProps {}

export const ThreadChat: FC<ThreadChatProps> = (props: ThreadChatProps) => {
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
                UserMessage: UserMessage,
                AssistantMessage: AssistantMessage,
              }}
            />
          </div>
        </ThreadPrimitive.Viewport>
      ) : null}
      <ThreadChatScrollToBottom />
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
              <SelectModel />
            </div>
            <div className="md:mb-8 mb-4">
              <SuggestedQuestions />
            </div>
            <ChatComposer
              submitDisabled={props.submitDisabled}
              messages={props.messages}
            />
          </div>
        </div>
      ) : (
        <ChatComposer
          submitDisabled={props.submitDisabled}
          messages={props.messages}
        />
      )}
    </ThreadPrimitive.Root>
  );
};

const ThreadChatScrollToBottom: FC = () => {
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
