import {
  useExternalMessageConverter,
  ThreadMessageLike,
  ToolCallContentPart,
} from "@assistant-ui/react";
import { AIMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";

// Not exposed by `@assistant-ui/react` package, but is
// the required return type for this callback function.
type Message =
  | ThreadMessageLike
  | {
      role: "tool";
      toolCallId: string;
      toolName?: string | undefined;
      result: any;
    };

export const convertLangchainMessages: useExternalMessageConverter.Callback<
  BaseMessage
> = (message): Message | Message[] => {
  if (typeof message.content !== "string") {
    throw new Error("Only text messages are supported");
  }

  switch (message._getType()) {
    case "system":
      return {
        role: "system",
        id: message.id,
        content: [{ type: "text", text: message.content }],
      };
    case "human":
      return {
        role: "user",
        id: message.id,
        content: [{ type: "text", text: message.content }],
      };
    case "ai":
      const aiMsg = message as AIMessage;
      const toolCallsContent: ToolCallContentPart[] = aiMsg.tool_calls?.length
        ? aiMsg.tool_calls.map((tc) => ({
            type: "tool-call" as const,
            toolCallId: tc.id ?? "",
            toolName: tc.name,
            args: tc.args,
            argsText: JSON.stringify(tc.args),
          }))
        : [];
      return {
        role: "assistant",
        id: message.id,
        content: [
          ...toolCallsContent,
          {
            type: "text",
            text: message.content,
          },
        ],
      };
    case "tool":
      return {
        role: "tool",
        toolName: message.name,
        toolCallId: (message as ToolMessage).tool_call_id,
        result: message.content,
      };
    default:
      throw new Error(`Unsupported message type: ${message._getType()}`);
  }
};

export function convertToOpenAIFormat(message: BaseMessage) {
  if (typeof message.content !== "string") {
    throw new Error("Only text messages are supported");
  }
  switch (message._getType()) {
    case "system":
      return {
        role: "system",
        content: message.content,
      };
    case "human":
      return {
        role: "user",
        content: message.content,
      };
    case "ai":
      return {
        role: "assistant",
        content: message.content,
      };
    case "tool":
      return {
        role: "tool",
        toolName: message.name,
        result: message.content,
      };
    default:
      throw new Error(`Unsupported message type: ${message._getType()}`);
  }
}
