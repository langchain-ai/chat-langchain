import { parsePartialJson } from "@langchain/core/output_parsers";
import { useEffect, useState } from "react";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { useToast } from "./use-toast";
import { v4 as uuidv4 } from "uuid";

import { Client } from "@langchain/langgraph-sdk";
import { getCookie, setCookie } from "../utils/cookies";
import { ThreadActual, useThreads } from "./useThreads";
import { ModelOptions } from "../types";
import { useRuns } from "./useRuns";
import { ASSISTANT_ID_COOKIE_NAME } from "../utils/constants";

export const createClient = () => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000/api";
  return new Client({
    apiUrl,
  });
};

const nodeToStep = (node: string) => {
  switch (node) {
    case "analyze_and_route_query":
      return 0;
    case "create_research_plan":
      return 1;
    case "conduct_research":
      return 2;
    case "respond":
      return 3;
    default:
      return 0;
  }
};

function addDocumentLinks(
  text: string,
  inputDocuments: Record<string, any>[],
): string {
  return text.replace(/\[(\d+)\]/g, (match, number) => {
    const index = parseInt(number, 10);
    if (index >= 0 && index < inputDocuments.length) {
      const document = inputDocuments[index];
      if (document && document.metadata && document.metadata.source) {
        return `[[${number}]](${document.metadata.source})`;
      }
    }
    // Return the original match if no corresponding document is found
    return match;
  });
}

export interface GraphInput {
  messages?: Record<string, any>[];
}

interface UseGraphInput {
  userId: string | undefined;
  threadId: string | undefined;
}

export function useGraph(inputArgs: UseGraphInput) {
  const { toast } = useToast();
  const { getThreadById, setThreadId } = useThreads(inputArgs.userId);
  const { shareRun } = useRuns();
  const [messages, setMessages] = useState<BaseMessage[]>([]);
  const [assistantId, setAssistantId] = useState<string>();
  const [selectedModel, setSelectedModel] =
    useState<ModelOptions>("openai/gpt-4o-mini");

  useEffect(() => {
    if (assistantId || typeof window === "undefined") return;
    getOrCreateAssistant();
  }, []);

  const getOrCreateAssistant = async () => {
    const assistantIdCookie = getCookie(ASSISTANT_ID_COOKIE_NAME);
    if (assistantIdCookie) {
      setAssistantId(assistantIdCookie);
      return;
    }
    const client = createClient();
    const assistant = await client.assistants.create({
      graphId: "chat",
    });
    setAssistantId(assistant.assistant_id);
    setCookie(ASSISTANT_ID_COOKIE_NAME, assistant.assistant_id);
  };

  const streamMessage = async (params: GraphInput) => {
    if (!inputArgs.threadId) {
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

    const stream = client.runs.stream(inputArgs.threadId, assistantId, {
      input,
      streamMode: "events",
      config: {
        configurable: {
          model_name: selectedModel,
        },
      },
    });
    let runId: string | undefined = undefined;
    let fullRoutingStr = "";
    let generatingQuestionsMessageId: string | undefined = undefined;
    let fullGeneratingQuestionsStr = "";
    const progressAIMessageId = uuidv4();
    let hasProgressBeenSet = false;

    for await (const chunk of stream) {
      if (!runId && chunk.data?.metadata?.run_id) {
        runId = chunk.data.metadata.run_id;
      }
      if (!hasProgressBeenSet) {
        setMessages((prevMessages) => {
          const existingMessageIndex = prevMessages.findIndex(
            (msg) => msg.id === progressAIMessageId,
          );

          if (existingMessageIndex !== -1) {
            return [
              ...prevMessages.slice(0, existingMessageIndex),
              new AIMessage({
                id: progressAIMessageId,
                content: "",
                tool_calls: [
                  {
                    name: "progress",
                    args: {
                      step: nodeToStep(chunk?.data?.metadata?.langgraph_node),
                    },
                  },
                ],
              }),
              ...prevMessages.slice(existingMessageIndex + 1),
            ];
          } else {
            console.warn(
              "Progress message ID is defined but not found in messages",
            );
            const newMessage = new AIMessage({
              id: progressAIMessageId,
              content: "",
              tool_calls: [
                {
                  name: "progress",
                  args: {
                    step: nodeToStep(chunk?.data?.metadata?.langgraph_node),
                  },
                },
              ],
            });
            return [...prevMessages, newMessage];
          }
        });
        hasProgressBeenSet = true;
      }

      if (chunk.data.event === "on_chain_start") {
        const node = chunk?.data?.metadata?.langgraph_node;
        if (
          [
            "analyze_and_route_query",
            "create_research_plan",
            "conduct_research",
            "respond",
          ].includes(node)
        ) {
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === progressAIMessageId,
            );

            if (existingMessageIndex !== -1) {
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  id: progressAIMessageId,
                  content: "",
                  tool_calls: [
                    {
                      name: "progress",
                      args: {
                        step: nodeToStep(node),
                      },
                    },
                  ],
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            } else {
              console.warn(
                "Progress message ID is defined but not found in messages",
              );
              return prevMessages;
            }
          });
        }

        if (node === "respond") {
          setMessages((prevMessages) => {
            const selectedDocumentsAIMessage = new AIMessage({
              content: "",
              tool_calls: [
                {
                  name: "selected_documents",
                  args: {
                    documents: chunk.data.data.input.documents,
                  },
                },
              ],
            });
            return [...prevMessages, selectedDocumentsAIMessage];
          });
        }
      }

      if (chunk.data.event === "on_chat_model_stream") {
        if (chunk.data.metadata.langgraph_node === "analyze_and_route_query") {
          const message = chunk.data.data.chunk;
          const toolCallChunk = message.tool_call_chunks?.[0];
          fullRoutingStr += toolCallChunk?.args || "";
          try {
            const parsedData: { logic: string } =
              parsePartialJson(fullRoutingStr);
            if (parsedData && parsedData.logic !== "") {
              setMessages((prevMessages) => {
                const existingMessageIndex = prevMessages.findIndex(
                  (msg) => msg.id === message.id,
                );

                if (existingMessageIndex !== -1) {
                  const newMessage = new AIMessage({
                    ...prevMessages[existingMessageIndex],
                    tool_calls: [
                      {
                        name: "router_logic",
                        args: parsedData,
                      },
                    ],
                  });

                  return [
                    ...prevMessages.slice(0, existingMessageIndex),
                    newMessage,
                    ...prevMessages.slice(existingMessageIndex + 1),
                  ];
                } else {
                  const newMessage = new AIMessage({
                    ...message,
                    tool_calls: [
                      {
                        name: "router_logic",
                        args: parsedData,
                      },
                    ],
                  });
                  return [...prevMessages, newMessage];
                }
              });
            }
          } catch (error) {
            console.error("Error parsing router logic data:", error);
          }
        }

        if (chunk.data.metadata.langgraph_node === "respond_to_general_query") {
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

        if (chunk.data.metadata.langgraph_node === "create_research_plan") {
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

        if (chunk.data.metadata.langgraph_node === "respond") {
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
      }

      if (chunk.data.event === "on_chain_end") {
        if (
          chunk.data.metadata.langgraph_node === "conduct_research" &&
          chunk.data.data?.output &&
          typeof chunk.data.data.output === "object" &&
          "question" in chunk.data.data.output
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
                        if (q.question === chunk.data.data.output.question) {
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

        if (
          ["respond", "respond_to_general_query", "ask_for_more_info"].includes(
            chunk?.data?.metadata?.langgraph_node,
          )
        ) {
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === progressAIMessageId,
            );
            if (existingMessageIndex !== -1) {
              // Create a new array with the updated message
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  id: progressAIMessageId,
                  content: "",
                  tool_calls: [
                    {
                      name: "progress",
                      args: {
                        step: 4,
                      },
                    },
                  ],
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            } else {
              console.warn(
                "Progress message ID is defined but not found in messages",
              );
              return prevMessages;
            }
          });
        }

        if (chunk.data.metadata.langgraph_node === "respond") {
          const inputDocuments = chunk.data.data.input.documents;
          const message = chunk.data.data.output.messages[0];
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (pMsg) => pMsg.id === message.id,
            );
            if (existingMessageIndex !== -1) {
              const newMessageWithLinks = new AIMessage({
                ...message,
                content: addDocumentLinks(message.content, inputDocuments),
              });

              return [
                ...prevMessages.slice(0, existingMessageIndex),
                newMessageWithLinks,
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            } else {
              return prevMessages;
            }
          });
        }
      }
    }

    if (runId) {
      // Chain `.then` to not block the stream
      shareRun(runId).then((sharedRunURL) => {
        if (sharedRunURL) {
          setMessages((prevMessages) => {
            const langSmithToolCallMessage = new AIMessage({
              content: "",
              id: uuidv4(),
              tool_calls: [
                {
                  name: "langsmith_tool_ui",
                  args: { sharedRunURL },
                  id: sharedRunURL
                    ?.split("https://smith.langchain.com/public/")[1]
                    .split("/")[0],
                },
              ],
            });
            return [...prevMessages, langSmithToolCallMessage];
          });
        }
      });
    }
  };

  const switchSelectedThread = async (thread: ThreadActual) => {
    setThreadId(thread.thread_id);
    if (!thread.values) {
      setMessages([]);
      return;
    }

    const actualMessages = (
      thread.values.messages as Record<string, any>[]
    ).flatMap((msg, index, array) => {
      if (msg.type === "human") {
        // insert progress bar afterwards
        const progressAIMessage = new AIMessage({
          id: uuidv4(),
          content: "",
          tool_calls: [
            {
              name: "progress",
              args: {
                step: 4, // Set to done.
              },
            },
          ],
        });
        return [
          new HumanMessage({
            ...msg,
            content: msg.content,
          }),
          progressAIMessage,
        ];
      }

      if (msg.type === "ai") {
        const isLastAiMessage =
          index === array.length - 1 || array[index + 1].type === "human";
        if (isLastAiMessage) {
          const routerMessage = thread.values?.router
            ? new AIMessage({
                content: "",
                id: uuidv4(),
                tool_calls: [
                  {
                    name: "router_logic",
                    args: thread.values.router,
                  },
                ],
              })
            : undefined;
          const selectedDocumentsAIMessage = thread.values?.documents?.length
            ? new AIMessage({
                content: "",
                id: uuidv4(),
                tool_calls: [
                  {
                    name: "selected_documents",
                    args: {
                      documents: thread.values.documents,
                    },
                  },
                ],
              })
            : undefined;
          const answerHeaderToolMsg = new AIMessage({
            content: "",
            tool_calls: [
              {
                name: "answer_header",
                args: {},
              },
            ],
          });
          return [
            ...(routerMessage ? [routerMessage] : []),
            ...(selectedDocumentsAIMessage ? [selectedDocumentsAIMessage] : []),
            answerHeaderToolMsg,
            new AIMessage({
              ...msg,
              content: msg.content,
            }),
          ];
        }
        return new AIMessage({
          ...msg,
          content: msg.content,
        });
      }

      return []; // Return an empty array for any other message types
    });

    setMessages(actualMessages);
  };

  return {
    messages,
    assistantId,
    selectedModel,
    setSelectedModel,
    setMessages,
    streamMessage,
    switchSelectedThread,
  };
}
