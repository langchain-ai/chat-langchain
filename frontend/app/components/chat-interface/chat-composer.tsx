"use client";

import { ComposerPrimitive, ThreadPrimitive } from "@assistant-ui/react";
import { type FC } from "react";

import { SendHorizontalIcon } from "lucide-react";
import { BaseMessage } from "@langchain/core/messages";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { cn } from "@/app/utils/cn";

export interface ChatComposerProps {
  messages: BaseMessage[];
  submitDisabled: boolean;
}

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

export const ChatComposer: FC<ChatComposerProps> = (
  props: ChatComposerProps,
) => {
  const isEmpty = props.messages.length === 0;

  return (
    <ComposerPrimitive.Root
      className={cn(
        // "bg-[#282828] focus-within:border-aui-ring/20 flex w-full items-center md:justify-left justify-center rounded-lg border px-2.5 py-2.5 shadow-sm transition-all duration-300 ease-in-out border-gray-600"
        "focus-within:border-aui-ring/20 flex w-full items-center md:justify-left justify-center rounded-2xl border px-2 py-2 shadow-sm transition-all duration-300 ease-in-out border-[#A5A5A5]",
        isEmpty ? "" : "md:ml-24 ml-3 mb-6",
        isEmpty ? "w-full" : "md:w-[70%] w-[95%] md:max-w-[832px]",
      )}
    >
      <ComposerPrimitive.Input
        autoFocus
        placeholder="Type your question here..."
        rows={1}
        className="placeholder:text-[#A5A5A5] text-black max-h-40 flex-1 resize-none border-none bg-transparent px-4 py-3 text-sm md:text-xl outline-none focus:ring-0 disabled:cursor-not-allowed"
      />
      <div className="flex-shrink-0">
        <ThreadPrimitive.If running={false} disabled={props.submitDisabled}>
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              tooltip="Send"
              variant="default"
              className="mr-2 my-1 size-8 p-2 bg-transparent text-black shadow-none transition-opacity ease-in"
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
