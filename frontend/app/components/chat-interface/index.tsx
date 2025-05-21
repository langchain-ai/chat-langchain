"use client";

import { ThreadPrimitive, ActionBarPrimitive } from "@assistant-ui/react";
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
import SignOut from "@/app/signout/SignOut";

export interface ThreadChatProps extends ChatComposerProps {}

export const ThreadChat: FC<ThreadChatProps> = (props: ThreadChatProps) => {
  const isEmpty = props.messages.length === 0;

  useGeneratingQuestionsUI();
  useAnswerHeaderToolUI();
  useProgressToolUI();
  useSelectedDocumentsUI();
  useRouterLogicUI();

  return (
    <ThreadPrimitive.Root className={cn("flex flex-col w-full bg-red-500 overflow-hidden", isEmpty ? "h-full" : "h-screen",)}>
      {!isEmpty ? (
        <ThreadPrimitive.Viewport
          className={cn(
            "overflow-hidden flex-1 overflow-y-auto scroll-smooth bg-inherit transition-all duration-300 ease-in-out w-full",
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
            <ActionBarPrimitive.Copy/>
          </div>
        </ThreadPrimitive.Viewport>
      ) : null}
      <ThreadChatScrollToBottom />
      {isEmpty ? (
        <div className="flex items-center justify-center flex-grow my-auto">
          <div className="flex flex-col items-center bg-orange-500 p-12 md:p-24 relative">
            <div className="absolute top-4 left-1/2 max-lg:-translate-x-1/2 lg:left-5 flex flex-row gap-4 m-4">
              <NextImage
                  src="/images/verafiles_banner.png"
                  className=""
                  alt="Verafiles Logo"
                  width={192}
                  height={192}
              />
              <div className="flex items-center text-[10px] md:text-xs text-black border border-[#D7D7D7] rounded-lg px-2">
                Claude 3.5 Haiku
              </div>
            </div>
            <div className="flex flex-col items-center justify-center">
              <p className="text-center text-black font-medium text-sm sm:text-lg m-1">SEEK helps you verify if something is true or not</p>
              <p className="text-center text-black font-bold text-3xl sm:text-4xl m-1">What would you like to know?</p>
              {/* <button className=""
              onClick={async()=> {
                await navigator.clipboard.writeText()
                
              }}>Copy</button> */}
            </div>
            <div className="my-4 sm:mt-8">
              <ChatComposer
                submitDisabled={props.submitDisabled}
                messages={props.messages}
              />
              <p className="text-black text-justify font-light text-xs md:text-sm mt-6">This tool answers your questions based on fact checks, 
              fact sheets and limited articles that VERA Files staff have written and edited. 
              This may not yet reflect our most recently published articles, 
              and may include responses based on the original time an article was published. 
              We strive to update our dataset with the most recent articles at the end of each day. 
              Read more about our tool here (insert link)</p>
            </div>
            <div className="w-full mt-2 md:mt-4">
              <h1 className="text-3xl text-black font-bold">
                Other Questions...
              </h1>
              <SuggestedQuestions />
            </div>
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
