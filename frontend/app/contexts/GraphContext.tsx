"use client";

import { parsePartialJson } from "@langchain/core/output_parsers";
import {
  createContext,
  Dispatch,
  ReactNode,
  SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { useToast } from "../hooks/use-toast";
import { v4 as uuidv4 } from "uuid";

import { useThreads } from "../hooks/useThreads";
import { ModelOptions } from "../types";
import { useRuns } from "../hooks/useRuns";
import { useUser } from "../hooks/useUser";
import { addDocumentLinks, nodeToStep } from "./utils";
import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { useStream } from "@langchain/langgraph-sdk/react";

export interface GraphInput {
  messages?: Record<string, any>[];
}

interface GraphData {
  runId: string;
  isStreaming: boolean;
  messages: BaseMessage[];
  selectedModel: ModelOptions;
  setSelectedModel: Dispatch<SetStateAction<ModelOptions>>;
  setMessages: Dispatch<SetStateAction<BaseMessage[]>>;
  streamMessage: (
    currentThreadId: string | null | undefined,
    params: GraphInput,
  ) => Promise<void>;
  switchSelectedThread: (thread: Thread) => void;
}

type UserDataContextType = ReturnType<typeof useUser>;

type ThreadsDataContextType = ReturnType<typeof useThreads>;

type GraphContentType = {
  graphData: GraphData;
  userData: UserDataContextType;
  threadsData: ThreadsDataContextType;
};

const GraphContext = createContext<GraphContentType | undefined>(undefined);

type StreamRunState = {
  progressMessageId: string;
  hasProgressBeenSet: boolean;
  fullRoutingStr: string;
  fullGeneratingQuestionsStr: string;
  generatingQuestionsMessageId?: string;
  runId?: string;
};

export function GraphProvider({ children }: { children: ReactNode }) {
  const { userId } = useUser();
  const {
    isUserThreadsLoading,
    userThreads,
    getThreadById,
    setUserThreads,
    getUserThreads,
    createThread,
    deleteThread,
  } = useThreads(userId);
  const [runId, setRunId] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [messages, setMessages] = useState<BaseMessage[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelOptions>(
    "openai/gpt-4.1-mini",
  );
  const [_threadId, setThreadId] = useQueryState("threadId");

  const streamStateRef = useRef<StreamRunState | null>(null);
  const runIdRef = useRef(runId);

  useEffect(() => {
    runIdRef.current = runId;
  }, [runId]);

  const { toast } = useToast();
  const { shareRun } = useRuns();

  const handleLangChainEvent = useCallback(
    (event: any) => {
      if (!event) return;

      const streamState = streamStateRef.current;
      const node = event?.metadata?.langgraph_node as string | undefined;

      if (event?.metadata?.run_id) {
        setRunId((prev) => prev || event.metadata.run_id);
        if (streamState) {
          streamState.runId = event.metadata.run_id;
        }
      }

      if (!streamState) {
        return;
      }

      const progressMessageId = streamState.progressMessageId;

      if (!streamState.hasProgressBeenSet) {
        const step = nodeToStep(node ?? "");
        const progressMessage = new AIMessage({
          id: progressMessageId,
          content: "",
          tool_calls: [
            {
              name: "progress",
              args: { step },
            },
          ],
        });
        setMessages((prevMessages) => {
          const existingMessageIndex = prevMessages.findIndex(
            (msg) => msg.id === progressMessageId,
          );

          if (existingMessageIndex !== -1) {
            return [
              ...prevMessages.slice(0, existingMessageIndex),
              progressMessage,
              ...prevMessages.slice(existingMessageIndex + 1),
            ];
          }

          return [...prevMessages, progressMessage];
        });
        streamState.hasProgressBeenSet = true;
      }

      if (event.event === "on_chain_start") {
        if (
          [
            "analyze_and_route_query",
            "create_research_plan",
            "conduct_research",
            "respond",
          ].includes(node ?? "")
        ) {
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === progressMessageId,
            );

            if (existingMessageIndex !== -1) {
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  id: progressMessageId,
                  content: "",
                  tool_calls: [
                    {
                      name: "progress",
                      args: {
                        step: nodeToStep(node ?? ""),
                      },
                    },
                  ],
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            }

            console.warn(
              "Progress message ID is defined but not found in messages",
            );
            return prevMessages;
          });
        }

        if (node === "respond") {
          const documents = event?.data?.input?.documents;
          if (documents?.length) {
            setMessages((prevMessages) => {
              const selectedDocumentsAIMessage = new AIMessage({
                content: "",
                tool_calls: [
                  {
                    name: "selected_documents",
                    args: {
                      documents,
                    },
                  },
                ],
              });
              return [...prevMessages, selectedDocumentsAIMessage];
            });
          }
        }
        return;
      }

      if (event.event === "on_chat_model_stream") {
        const message = event?.data?.chunk;
        if (!message) return;

        if (node === "analyze_and_route_query") {
          const toolCallChunk = message.tool_call_chunks?.[0];
          streamState.fullRoutingStr += toolCallChunk?.args || "";
          try {
            const parsedData: { logic: string } = parsePartialJson(
              streamState.fullRoutingStr,
            );
            if (parsedData && parsedData.logic !== "") {
              setMessages((prevMessages) => {
                const existingMessageIndex = prevMessages.findIndex(
                  (msg) => msg.id === message.id,
                );

                const routerMessage = new AIMessage({
                  ...message,
                  tool_calls: [
                    {
                      name: "router_logic",
                      args: parsedData,
                    },
                  ],
                });

                if (existingMessageIndex !== -1) {
                  return [
                    ...prevMessages.slice(0, existingMessageIndex),
                    routerMessage,
                    ...prevMessages.slice(existingMessageIndex + 1),
                  ];
                }

                return [...prevMessages, routerMessage];
              });
            }
          } catch (error) {
            console.error("Error parsing router logic data:", error);
          }
        }

        if (node === "respond_to_general_query") {
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === message.id,
            );
            if (existingMessageIndex !== -1) {
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  ...prevMessages[existingMessageIndex],
                  content:
                    (prevMessages[existingMessageIndex].content as string) +
                    message.content,
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            }

            const newMessage = new AIMessage({
              ...message,
            });
            return [...prevMessages, newMessage];
          });
        }

        if (node === "create_research_plan") {
          streamState.generatingQuestionsMessageId = message.id;
          const toolCallChunk = message.tool_call_chunks?.[0];
          streamState.fullGeneratingQuestionsStr += toolCallChunk?.args || "";
          try {
            const parsedData: { steps: string[] } = parsePartialJson(
              streamState.fullGeneratingQuestionsStr,
            );
            if (parsedData && Array.isArray(parsedData.steps)) {
              const questions = parsedData.steps
                .map((step, index) => ({
                  step: index + 1,
                  question: step.trim(),
                }))
                .filter((q) => q.question !== "");

              if (questions.length > 0) {
                setMessages((prevMessages) => {
                  const existingMessageIndex = prevMessages.findIndex(
                    (msg) => msg.id === message.id,
                  );

                  const toolCall = {
                    name: "generating_questions",
                    args: { questions },
                  };

                  if (existingMessageIndex !== -1) {
                    return [
                      ...prevMessages.slice(0, existingMessageIndex),
                      new AIMessage({
                        ...prevMessages[existingMessageIndex],
                        content: "",
                        tool_calls: [toolCall],
                      }),
                      ...prevMessages.slice(existingMessageIndex + 1),
                    ];
                  }

                  const newMessage = new AIMessage({
                    ...message,
                    content: "",
                    tool_calls: [toolCall],
                  });
                  return [...prevMessages, newMessage];
                });
              }
            }
          } catch (error) {
            console.error("Error parsing generating questions data:", error);
          }
        }

        if (node === "respond") {
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === message.id,
            );
            if (existingMessageIndex !== -1) {
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  ...prevMessages[existingMessageIndex],
                  content:
                    (prevMessages[existingMessageIndex].content as string) +
                    message.content,
                }),
                ...prevMessages.slice(existingMessageIndex + 1),
              ];
            }

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
          });
        }
        return;
      }

      if (event.event === "on_chain_end") {
        if (
          node === "conduct_research" &&
          event?.data?.output &&
          typeof event.data.output === "object" &&
          "question" in event.data.output
        ) {
          setMessages((prevMessages) => {
            const generatingId = streamState.generatingQuestionsMessageId;
            if (!generatingId) {
              return prevMessages;
            }
            const foundIndex = prevMessages.findIndex(
              (msg) =>
                "tool_calls" in msg &&
                Array.isArray((msg as AIMessage).tool_calls) &&
                (msg as AIMessage).tool_calls?.length > 0 &&
                (msg as AIMessage).tool_calls?.[0].name ===
                  "generating_questions" &&
                msg.id === generatingId,
            );

            if (foundIndex === -1) {
              return prevMessages;
            }

            const messageToUpdate = prevMessages[foundIndex] as AIMessage;
            const updatedToolCalls = messageToUpdate.tool_calls?.map(
              (toolCall) => {
                if (toolCall.name === "generating_questions") {
                  const updatedQuestions = toolCall.args.questions.map(
                    (q: any) => {
                      if (q.question === event.data.output.question) {
                        return {
                          ...q,
                          queries: event.data.output.queries,
                          documents: event.data.output.documents,
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
          });
        }

        if (
          ["respond", "respond_to_general_query", "ask_for_more_info"].includes(
            node ?? "",
          )
        ) {
          setMessages((prevMessages) => {
            const existingMessageIndex = prevMessages.findIndex(
              (msg) => msg.id === progressMessageId,
            );
            if (existingMessageIndex !== -1) {
              return [
                ...prevMessages.slice(0, existingMessageIndex),
                new AIMessage({
                  id: progressMessageId,
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
            }

            console.warn(
              "Progress message ID is defined but not found in messages",
            );
            return prevMessages;
          });
        }

        if (node === "respond") {
          const inputDocuments = event?.data?.input?.documents;
          const modelMessage = event?.data?.output?.messages?.[0];
          if (modelMessage && inputDocuments) {
            setMessages((prevMessages) => {
              const existingMessageIndex = prevMessages.findIndex(
                (pMsg) => pMsg.id === modelMessage.id,
              );
              if (existingMessageIndex !== -1) {
                const newMessageWithLinks = new AIMessage({
                  ...modelMessage,
                  content: addDocumentLinks(
                    modelMessage.content,
                    inputDocuments,
                  ),
                });

                return [
                  ...prevMessages.slice(0, existingMessageIndex),
                  newMessageWithLinks,
                  ...prevMessages.slice(existingMessageIndex + 1),
                ];
              }
              return prevMessages;
            });
          }
        }
      }
    },
    [setMessages],
  );

  const thread = useStream<
    Record<string, unknown>,
    {
      ConfigurableType: { query_model: string; response_model: string };
    }
  >({
    apiUrl: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000/api",
    assistantId: process.env.NEXT_PUBLIC_ASSISTANT_ID ?? "chat",
    threadId: _threadId ?? undefined,
    onThreadId: (tid) => setThreadId(tid ?? null),
    messagesKey: "messages",
    reconnectOnMount: true,
    onMetadataEvent: (meta) => {
      if (meta?.run_id) {
        setRunId(meta.run_id);
        if (streamStateRef.current) {
          streamStateRef.current.runId = meta.run_id;
        }
      }
    },
    onLangChainEvent: handleLangChainEvent,
    onFinish: async (_state, meta) => {
      setIsStreaming(false);
      const currentState = streamStateRef.current;
      streamStateRef.current = null;
      const finalRunId =
        meta?.run_id ?? currentState?.runId ?? runIdRef.current;
      if (!finalRunId) return;

      setRunId(finalRunId);
      try {
        const sharedRunURL = await shareRun(finalRunId);
        if (!sharedRunURL) return;

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
                  ?.split("/")?.[0],
              },
            ],
          });
          return [...prevMessages, langSmithToolCallMessage];
        });
      } catch (error) {
        console.error("Failed to share run", error);
      }
    },
    onError: (error) => {
      console.error(error);
      streamStateRef.current = null;
      setIsStreaming(false);
      toast({
        title: "Error",
        description: "Streaming failed",
      });
    },
  });

  const streamMessage = useCallback(
    async (
      currentThreadId: string | null | undefined,
      params: GraphInput,
    ): Promise<void> => {
      if (!userId) {
        toast({
          title: "Error",
          description: "User ID not found",
        });
        return;
      }

      const filteredMessages = (params.messages ?? []).filter((msg) => {
        if (msg.role !== "assistant") {
          return true;
        }
        const toolCalls = (msg as any).tool_calls as
          | Array<{ name: string }>
          | undefined;
        return !(
          toolCalls && toolCalls.some((tc) => tc.name === "artifact_ui")
        );
      });

      setRunId("");
      setIsStreaming(true);
      streamStateRef.current = {
        progressMessageId: uuidv4(),
        hasProgressBeenSet: false,
        fullRoutingStr: "",
        fullGeneratingQuestionsStr: "",
      };

      try {
        await thread.submit(
          { messages: filteredMessages },
          {
            threadId: currentThreadId || undefined,
            config: {
              configurable: {
                query_model: selectedModel,
                response_model: selectedModel,
              },
            },
            metadata: userId ? { user_id: userId } : undefined,
            streamResumable: true,
            onDisconnect: "continue",
            streamMode: ["events"],
          },
        );
      } catch (error) {
        console.error(error);
        streamStateRef.current = null;
        setIsStreaming(false);
        toast({
          title: "Error",
          description: "Streaming failed",
        });
      }
    },
    [userId, toast, thread, selectedModel],
  );

  const switchSelectedThread = useCallback(
    (thread: Thread) => {
      setThreadId(thread.thread_id);
      streamStateRef.current = null;
      setIsStreaming(false);

      if (!thread.values) {
        setMessages([]);
        return;
      }

      const threadValues = thread.values as Record<string, any>;

      const actualMessages = (
        (threadValues.messages as Record<string, any>[]) || []
      ).flatMap((msg, index, array) => {
        if (msg.type === "human") {
          const progressAIMessage = new AIMessage({
            id: uuidv4(),
            content: "",
            tool_calls: [
              {
                name: "progress",
                args: {
                  step: 4,
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
            const routerMessage = threadValues.router
              ? new AIMessage({
                  content: "",
                  id: uuidv4(),
                  tool_calls: [
                    {
                      name: "router_logic",
                      args: threadValues.router,
                    },
                  ],
                })
              : undefined;
            const selectedDocumentsAIMessage = threadValues.documents?.length
              ? new AIMessage({
                  content: "",
                  id: uuidv4(),
                  tool_calls: [
                    {
                      name: "selected_documents",
                      args: {
                        documents: threadValues.documents,
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
              ...(selectedDocumentsAIMessage
                ? [selectedDocumentsAIMessage]
                : []),
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

        return [];
      });

      setMessages(actualMessages);
    },
    [setThreadId],
  );

  const contextValue: GraphContentType = useMemo(
    () => ({
      userData: {
        userId,
      },
      threadsData: {
        isUserThreadsLoading,
        userThreads,
        getThreadById,
        setUserThreads,
        getUserThreads,
        createThread,
        deleteThread,
      },
      graphData: {
        runId,
        isStreaming,
        messages,
        selectedModel,
        setSelectedModel,
        setMessages,
        streamMessage,
        switchSelectedThread,
      },
    }),
    [
      userId,
      isUserThreadsLoading,
      userThreads,
      getThreadById,
      setUserThreads,
      getUserThreads,
      createThread,
      deleteThread,
      runId,
      isStreaming,
      messages,
      selectedModel,
      streamMessage,
      switchSelectedThread,
    ],
  );

  return (
    <GraphContext.Provider value={contextValue}>
      {children}
    </GraphContext.Provider>
  );
}

export function useGraphContext() {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error("useGraphContext must be used within a GraphProvider");
  }
  return context;
}
