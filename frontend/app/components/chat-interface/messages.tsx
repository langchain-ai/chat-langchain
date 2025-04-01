"use client";

import {
  MessagePrimitive,
  useMessage,
  useThreadRuntime,
} from "@assistant-ui/react";
import { useState, type FC } from "react";

import { MarkdownText } from "../ui/assistant-ui/markdown-text";
import { useGraphContext } from "@/app/contexts/GraphContext";
import { useRuns } from "@/app/hooks/useRuns";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { ThumbsDownIcon, ThumbsUpIcon } from "lucide-react";

export const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="pt-2 sm:pt-4 flex w-full md:max-w-4xl md:mx-0 mx-auto max-w-[95%] md:py-4 py-2">
      <div className="bg-inherit text-white break-words rounded-2xl sm:rounded-3xl pt-2 md:pt-2.5 mb-[-15px] sm:mb-[-25px] text-2xl sm:text-4xl font-light">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

function FeedbackButtons() {
  const {
    graphData: { runId, isStreaming },
  } = useGraphContext();
  const { sendFeedback } = useRuns();
  const [feedback, setFeedback] = useState<"good" | "bad">();

  const feedbackKey = "user_feedback";
  const goodScore = 1;
  const badScore = 0;

  if (!runId || isStreaming) return null;

  if (feedback) {
    return (
      <div className="flex gap-2 items-center mt-4">
        {feedback === "good" ? (
          <ThumbsUpIcon className="w-4 h-4 text-green-500" />
        ) : (
          <ThumbsDownIcon className="w-4 h-4 text-red-500" />
        )}
      </div>
    );
  }

  return (
    <div className="flex gap-2 items-center mt-4">
      <TooltipIconButton
        delayDuration={200}
        variant="ghost"
        tooltip="Good response"
        onClick={() => {
          sendFeedback(runId, feedbackKey, goodScore);
          setFeedback("good");
        }}
      >
        <ThumbsUpIcon className="w-4 h-4" />
      </TooltipIconButton>
      <TooltipIconButton
        delayDuration={200}
        variant="ghost"
        tooltip="Bad response"
        onClick={() => {
          sendFeedback(runId, feedbackKey, badScore);
          setFeedback("bad");
        }}
      >
        <ThumbsDownIcon className="w-4 h-4" />
      </TooltipIconButton>
    </div>
  );
}

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
        {isLast && <FeedbackButtons />}
      </div>
    </MessagePrimitive.Root>
  );
};
