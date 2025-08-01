"use client";

import {
  MessagePrimitive,
  useMessage,
  useThreadRuntime,
} from "@assistant-ui/react";
import { useState, type FC } from "react";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";

import { MarkdownText } from "../ui/assistant-ui/markdown-text";
import { useGraphContext } from "@/app/contexts/GraphContext";
import { useRuns } from "@/app/hooks/useRuns";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { ThumbsDownIcon, ThumbsUpIcon } from "lucide-react";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "../ui/collapsible";

// Helper function to generate a brief headline from long text
function generateHeadline(text: string, maxLength: number = 60): string {
  if (text.length <= maxLength) return text;

  // Try to find the first sentence or logical break
  const firstSentence = text.match(/^[^.!?]*[.!?]/);
  if (firstSentence && firstSentence[0].length <= maxLength) {
    return firstSentence[0].trim();
  }

  // Try to find subject line if it looks like an email
  const subjectMatch = text.match(/^Subject:\s*([^\n\r]+)/i);
  if (subjectMatch && subjectMatch[1].length <= maxLength) {
    return subjectMatch[1].trim();
  }

  // Fallback to truncated text at word boundary
  const truncated = text.substring(0, maxLength);
  const lastSpace = truncated.lastIndexOf(" ");
  return lastSpace > maxLength * 0.7
    ? truncated.substring(0, lastSpace) + "..."
    : truncated + "...";
}

export const UserMessage: FC = () => {
  const message = useMessage();
  const [isExpanded, setIsExpanded] = useState(false);

  // Get the text content from the message
  const textContent =
    message.content?.[0]?.type === "text" ? message.content[0].text : "";
  const isLongText = textContent.length > 150; // Threshold for collapsible behavior

  if (!isLongText) {
    // For short text, use the original styling
    return (
      <MessagePrimitive.Root className="pt-2 sm:pt-4 flex w-full md:max-w-4xl md:mx-0 mx-auto max-w-[95%] md:py-4 py-2">
        <div className="bg-inherit text-white break-words rounded-2xl sm:rounded-3xl pt-2 md:pt-2.5 mb-[-15px] sm:mb-[-25px] text-2xl sm:text-4xl font-light">
          <MessagePrimitive.Content />
        </div>
      </MessagePrimitive.Root>
    );
  }

  const headline = generateHeadline(textContent);

  return (
    <MessagePrimitive.Root className="pt-2 sm:pt-4 flex w-full md:max-w-4xl md:mx-0 mx-auto max-w-[95%] md:py-4 py-2">
      <Collapsible
        open={isExpanded}
        onOpenChange={setIsExpanded}
        className="w-full"
      >
        <CollapsibleTrigger className="w-full text-left group hover:opacity-80 transition-opacity">
          <div className="bg-inherit text-white break-words rounded-2xl sm:rounded-3xl pt-2 md:pt-2.5 mb-[-15px] sm:mb-[-25px] flex items-start gap-2">
            <div className="text-2xl sm:text-4xl font-light flex-1">
              {headline}
            </div>
            <div className="flex items-center mt-1 text-gray-400 group-hover:text-white transition-colors">
              {isExpanded ? (
                <ChevronDownIcon className="w-5 h-5 sm:w-6 sm:h-6" />
              ) : (
                <ChevronRightIcon className="w-5 h-5 sm:w-6 sm:h-6" />
              )}
            </div>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent className="overflow-hidden">
          <div className="pt-4 pb-2">
            <div className="bg-gray-800/50 text-gray-200 text-sm sm:text-base rounded-lg p-3 sm:p-4 border border-gray-700/50">
              <div className="font-medium text-gray-300 mb-2 text-xs sm:text-sm uppercase tracking-wide">
                Full Message
              </div>
              <div className="whitespace-pre-wrap break-words leading-relaxed">
                {textContent}
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
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
