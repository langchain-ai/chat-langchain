import {
  useExternalMessageConverter,
  ThreadMessageLike,
  ToolCallContentPart,
} from "@assistant-ui/react";
import type { Message } from "@langchain/langgraph-sdk";

// Not exposed by `@assistant-ui/react` package, but is
// the required return type for this callback function.
type AssistantUiMessage =
  | ThreadMessageLike
  | {
      role: "tool";
      toolCallId: string;
      toolName?: string | undefined;
      result: any;
    };

export function messageContentToText(message: Message): string {
  const { content } = message;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const textParts = content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (part?.type === "text" && typeof part.text === "string") {
          return part.text;
        }
        return "";
      })
      .filter(Boolean);
    return textParts.join("\n");
  }
  return "";
}

export const convertLangchainMessages: useExternalMessageConverter.Callback<
  Message
> = (message): AssistantUiMessage | AssistantUiMessage[] => {
  const textContent = messageContentToText(message);

  switch (message.type) {
    case "system":
      return {
        role: "system",
        id: message.id,
        content: [{ type: "text", text: textContent }],
      };
    case "human":
      return {
        role: "user",
        id: message.id,
        content: [{ type: "text", text: textContent }],
      };
    case "ai":
      const toolCallsContent: ToolCallContentPart[] = Array.isArray(
        message.tool_calls,
      )
        ? message.tool_calls.map((toolCall) => ({
            type: "tool-call" as const,
            toolCallId: toolCall.id ?? "",
            toolName: toolCall.name,
            args: toolCall.args,
            argsText: JSON.stringify(toolCall.args),
          }))
        : [];
      return {
        role: "assistant",
        id: message.id,
        content: [
          ...toolCallsContent,
          {
            type: "text",
            text: textContent,
          },
        ],
      };
    case "tool":
      return {
        role: "tool",
        toolName: message.name,
        toolCallId: message.tool_call_id,
        result: message.content,
      };
    default:
      throw new Error(`Unsupported message type: ${message.type}`);
  }
};

export function convertToOpenAIFormat(message: Message) {
  const textContent = messageContentToText(message);

  switch (message.type) {
    case "system":
      return {
        role: "system",
        content: textContent,
      };
    case "human":
      return {
        role: "user",
        content: textContent,
      };
    case "ai":
      return {
        role: "assistant",
        content: textContent,
      };
    case "tool":
      return {
        role: "tool",
        toolName: message.name,
        result: message.content,
      };
    default:
      throw new Error(`Unsupported message type: ${message.type}`);
  }
}
