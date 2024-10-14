import { parsePartialJson } from "@langchain/core/output_parsers";
import { useEffect, useState } from "react";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { useToast } from "./use-toast";

import { Client } from "@langchain/langgraph-sdk";
import { getCookie, setCookie } from "../utils/cookies";

export const createClient = () => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000/api";
  return new Client({
    apiUrl,
  });
};

export interface GraphInput {
  messages?: Record<string, any>[];
}

export function useGraph() {
  const { toast } = useToast();
  const [messages, setMessages] = useState<BaseMessage[]>([]);
  const [assistantId, setAssistantId] = useState<string>();
  const [threadId, setThreadId] = useState<string | null>(null);

  useEffect(() => {
    if (threadId || typeof window === "undefined") return;
    createThread();
  }, []);

  useEffect(() => {
    if (assistantId || typeof window === "undefined") return;
    getOrCreateThread();
  }, []);

  const createThread = async () => {
    setMessages([]);
    const client = createClient();
    let thread;
    try {
      thread = await client.threads.create();
      if (!thread || !thread.thread_id) {
        throw new Error("Thread creation failed.");
      }
      setThreadId(thread.thread_id);
    } catch (e) {
      console.error("Error creating thread", e);
      toast({
        title: "Error creating thread.",
      });
    }
    return thread;
  };

  const getOrCreateThread = async () => {
    const assistantIdCookie = getCookie("oc_assistant_id");
    if (assistantIdCookie) {
      setAssistantId(assistantIdCookie);
      return;
    }
    const client = createClient();
    const assistant = await client.assistants.create({
      graphId: "agent",
    });
    setAssistantId(assistant.assistant_id);
    setCookie("oc_assistant_id", assistant.assistant_id);
  };

  const streamMessage = async (params: GraphInput) => {
    if (!threadId) {
      toast({
        title: "Error",
        description: "Thread ID not found",
      });
      return undefined;
    }
    if (!assistantId) {
      toast({
        title: "Error",
        description: "Assistant ID not found",
      });
      return undefined;
    }

    const client = createClient();

    const input = {
      messages: params.messages?.filter((msg) => {
        if (msg.role !== "assistant") {
          return true;
        }
        const aiMsg = msg as AIMessage;
        // Filter our artifact ui tool calls from going to the server.
        if (
          aiMsg.tool_calls &&
          aiMsg.tool_calls.some((tc) => tc.name === "artifact_ui")
        ) {
          return false;
        }
        return true;
      }),
    };

    const stream = client.runs.stream(threadId, assistantId, {
      input,
      streamMode: "events",
    });
    let runId: string | undefined = undefined;
    let generatingQuestionsMessageId: string | undefined = undefined;
    let fullGeneratingQuestionsStr = "";

    for await (const chunk of stream) {
      if (!runId && chunk.data?.metadata?.run_id) {
        runId = chunk.data.metadata.run_id;
      }

      if (chunk.data.event === "on_chat_model_stream") {
        if (chunk.data.metadata.langgraph_node === "general") {
          const message = chunk.data.data.chunk;
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === message.id,
            );
            if (existingMessageIndex !== -1) {
              // Create a new array with the updated message
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  ...prevMessages[existingMessageIndex],
                  content:
                    prevMessages[existingMessageIndex].content +
                    message.content,
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            } else {
              const newMessage = new AIMessage({
                ...message,
              });
              return [...prevMessages, newMessage];
            }
          });
        }

        // if (chunk.data.metadata.langgraph_node === "generate_questions") {
        //   const message = chunk.data.data.chunk;
        //   generatingQuestionsMessageId = message.id;
        //   const toolCallChunk = message.tool_call_chunks?.[0];
        //   fullGeneratingQuestionsStr += toolCallChunk?.args || "";
        //   try {
        //     const parsedData: { steps: string[] } = parsePartialJson(
        //       fullGeneratingQuestionsStr,
        //     );
        //     if (parsedData && Array.isArray(parsedData.steps)) {
        //       setMessages((prevMessages) => {
        //         const existingMessageIndex = prevMessages.findIndex(
        //           (msg) => msg.id === message.id,
        //         );
        //         if (existingMessageIndex !== -1) {
        //           const existingMessage = prevMessages[
        //             existingMessageIndex
        //           ] as AIMessage;
        //           const existingToolCalls = existingMessage.tool_calls || [];

        //           const updatedToolCalls = parsedData.steps.flatMap(
        //             (step, index) => {
        //               const existingToolCall = existingToolCalls.find(
        //                 (tc) => tc.args.step === index + 1,
        //               );
        //               if (existingToolCall) {
        //                 // Update existing tool call
        //                 return {
        //                   ...existingToolCall,
        //                   args: {
        //                     ...existingToolCall.args,
        //                     question: step.trim(),
        //                   },
        //                 };
        //               } else if (step && step.trim() !== "") {
        //                 // Create new tool call
        //                 return {
        //                   name: "generating_questions",
        //                   args: { step: index + 1, question: step.trim() },
        //                 };
        //               }
        //               // Return empty array because we're using .flatMap
        //               return [];
        //             },
        //           );

        //           if (updatedToolCalls.length > 0) {
        //             return [
        //               ...prevMessages.slice(0, existingMessageIndex),
        //               new AIMessage({
        //                 ...existingMessage,
        //                 content: "",
        //                 tool_calls: updatedToolCalls,
        //               }),
        //               ...prevMessages.slice(existingMessageIndex + 1),
        //             ];
        //           }
        //         } else if (
        //           parsedData.steps.some((step) => step && step.trim() !== "")
        //         ) {
        //           const newToolCalls = parsedData.steps
        //             .map((step, index) => {
        //               if (step && step.trim() !== "") {
        //                 return {
        //                   name: "generating_questions",
        //                   args: { step: index + 1, question: step.trim() },
        //                 };
        //               }
        //               return null;
        //             })
        //             .filter(Boolean);

        //           const newMessage = new AIMessage({
        //             ...message,
        //             content: "",
        //             tool_calls: newToolCalls,
        //           });
        //           return [...prevMessages, newMessage];
        //         }
        //         return prevMessages;
        //       });
        //     }
        //   } catch (error) {
        //     console.error("Error parsing generating questions data:", error);
        //   }
        // }
        if (chunk.data.metadata.langgraph_node === "generate_questions") {
          const message = chunk.data.data.chunk;
          generatingQuestionsMessageId = message.id;
          const toolCallChunk = message.tool_call_chunks?.[0];
          fullGeneratingQuestionsStr += toolCallChunk?.args || "";
          try {
            const parsedData: { steps: string[] } = parsePartialJson(
              fullGeneratingQuestionsStr,
            );
            if (parsedData && Array.isArray(parsedData.steps)) {
              setMessages((prevMessages) => {
                const existingMessageIndex = prevMessages.findIndex(
                  (msg) => msg.id === message.id,
                );

                const questions = parsedData.steps
                  .map((step, index) => ({
                    step: index + 1,
                    question: step.trim(),
                  }))
                  .filter((q) => q.question !== "");

                if (existingMessageIndex !== -1) {
                  const existingMessage = prevMessages[
                    existingMessageIndex
                  ] as AIMessage;
                  const existingToolCalls = existingMessage.tool_calls || [];

                  let updatedToolCall;
                  if (existingToolCalls[0].name === "generating_questions") {
                    // Update existing tool call
                    updatedToolCall = {
                      ...existingToolCalls[0],
                      args: {
                        questions,
                      },
                    };
                  } else {
                    // Create new tool call
                    updatedToolCall = {
                      name: "generating_questions",
                      args: { questions },
                    };
                  }

                  return [
                    ...prevMessages.slice(0, existingMessageIndex),
                    new AIMessage({
                      ...existingMessage,
                      content: "",
                      tool_calls: [updatedToolCall],
                    }),
                    ...prevMessages.slice(existingMessageIndex + 1),
                  ];
                } else if (questions.length > 0) {
                  // Create new message with tool call
                  const newToolCall = {
                    name: "generating_questions",
                    args: { questions },
                  };

                  const newMessage = new AIMessage({
                    ...message,
                    content: "",
                    tool_calls: [newToolCall],
                  });
                  return [...prevMessages, newMessage];
                }
                return prevMessages;
              });
            }
          } catch (error) {
            console.error("Error parsing generating questions data:", error);
          }
        }

        if (chunk.data.metadata.langgraph_node === "generate") {
          const message = chunk.data.data.chunk;
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === message.id,
            );
            if (existingMessageIndex !== -1) {
              // Create a new array with the updated message
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  ...prevMessages[existingMessageIndex],
                  content:
                    prevMessages[existingMessageIndex].content +
                    message.content,
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            } else {
              const answerHeaderToolMsg = new AIMessage({
                content: "",
                tool_calls: [
                  {
                    name: "answer_header",
                    args: {},
                  },
                ],
              });
              const newMessage = new AIMessage({
                ...message,
              });
              return [...prevMessages, answerHeaderToolMsg, newMessage];
            }
          });
        }
      } else if (chunk.data.event === "on_chain_end") {
        if (
          chunk.data.metadata.langgraph_node === "research_node" &&
          chunk.data.data?.output &&
          typeof chunk.data.data.output === "object" &&
          "sub_question" in chunk.data.data.output
        ) {
          setMessages((prevMessages) => {
            const foundIndex = prevMessages.findIndex(
              (msg) =>
                "tool_calls" in msg &&
                Array.isArray(msg.tool_calls) &&
                msg.tool_calls.length > 0 &&
                msg.tool_calls[0].name === "generating_questions" &&
                msg.id === generatingQuestionsMessageId,
            );

            if (foundIndex !== -1) {
              const messageToUpdate = prevMessages[foundIndex] as AIMessage;
              const updatedToolCalls = messageToUpdate.tool_calls?.map(
                (toolCall) => {
                  if (toolCall.name === "generating_questions") {
                    const updatedQuestions = toolCall.args.questions.map(
                      (q: any) => {
                        if (
                          q.question === chunk.data.data.output.sub_question
                        ) {
                          return {
                            ...q,
                            queries: chunk.data.data.output.queries,
                            documents: chunk.data.data.output.documents,
                          };
                        }
                        return q;
                      },
                    );

                    return {
                      ...toolCall,
                      args: {
                        ...toolCall.args,
                        questions: updatedQuestions,
                      },
                    };
                  }
                  return toolCall;
                },
              );

              const updatedMessage = new AIMessage({
                ...messageToUpdate,
                tool_calls: updatedToolCalls,
              });

              return [
                ...prevMessages.slice(0, foundIndex),
                updatedMessage,
                ...prevMessages.slice(foundIndex + 1),
              ];
            }

            // Return the previous messages unchanged if no matching message found
            return prevMessages;
          });
        }
      }
    }
  };

  return {
    messages,
    assistantId,
    setMessages,
    streamMessage,
    createThread,
  };
}
