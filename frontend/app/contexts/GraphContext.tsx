"use client";

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
import { useToast } from "../hooks/use-toast";
import { v4 as uuidv4 } from "uuid";

import { useThreads } from "../hooks/useThreads";
import { ModelOptions } from "../types";
import { useRuns } from "../hooks/useRuns";
import { useUser } from "../hooks/useUser";
import { addDocumentLinks, nodeToStep } from "./utils";
import type { Message, Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { useStream } from "@langchain/langgraph-sdk/react";

export interface GraphInput {
  messages?: Record<string, any>[];
}

interface GraphData {
  runId: string;
  isStreaming: boolean;
  messages: Message[];
  selectedModel: ModelOptions;
  setSelectedModel: Dispatch<SetStateAction<ModelOptions>>;
  setMessages: Dispatch<SetStateAction<Message[]>>;
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
  generatingQuestionsMessageId?: string;
  routerMessageId?: string;
  selectedDocumentsMessageId?: string;
  answerHeaderInserted?: boolean;
  latestDocuments?: Record<string, any>[];
  runId?: string;
};

function flattenMessageContent(content: Message["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (part?.type === "text" && typeof part.text === "string") {
          return part.text;
        }
        return "";
      })
      .join("")
      .trim();
  }

  return "";
}

function isToolMessageWithName(message: Message, toolName: string): boolean {
  return (
    message.type === "ai" &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.some((toolCall) => toolCall.name === toolName)
  );
}

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
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedModel, setSelectedModel] =
    useState<ModelOptions>("openai/gpt-5-mini");
  const [_threadId, setThreadId] = useQueryState("threadId");

  const streamStateRef = useRef<StreamRunState | null>(null);
  const runIdRef = useRef(runId);

  useEffect(() => {
    runIdRef.current = runId;
  }, [runId]);

  const { toast } = useToast();
  const { shareRun } = useRuns();

  const updateProgress = useCallback((step: number) => {
    const streamState = streamStateRef.current;
    if (!streamState) {
      return;
    }

    const progressMessageId = streamState.progressMessageId;
    const progressMessage: Message = {
      type: "ai",
      id: progressMessageId,
      content: "",
      tool_calls: [
        {
          name: "progress",
          args: { step },
        },
      ],
    };

    setMessages((prevMessages) => {
      const existingMessageIndex = prevMessages.findIndex(
        (msg) => msg.id === progressMessageId,
      );

      if (existingMessageIndex !== -1) {
        const nextMessages = [...prevMessages];
        nextMessages[existingMessageIndex] = progressMessage;
        return nextMessages;
      }

      return [...prevMessages, progressMessage];
    });

    streamState.hasProgressBeenSet = true;
  }, []);

  const handleUpdateEvent = useCallback(
    (data: Record<string, any> | undefined) => {
      if (!data) return;
      const streamState = streamStateRef.current;
      if (!streamState) return;

      for (const [node, update] of Object.entries(data)) {
        if (!update || typeof update !== "object") continue;

        if (
          node === "analyze_and_route_query" ||
          node === "create_research_plan" ||
          node === "conduct_research" ||
          node === "respond"
        ) {
          updateProgress(nodeToStep(node));
        } else if (
          node === "respond_to_general_query" ||
          node === "ask_for_more_info"
        ) {
          updateProgress(4);
        }

        if (node === "analyze_and_route_query" && update.router) {
          const routerMessageId = streamState.routerMessageId ?? uuidv4();
          streamState.routerMessageId = routerMessageId;
          const routerMessage: Message = {
            type: "ai",
            id: routerMessageId,
            content: "",
            tool_calls: [
              {
                name: "router_logic",
                args: update.router,
              },
            ],
          };

          setMessages((prev) => {
            const idx = prev.findIndex((msg) => msg.id === routerMessageId);
            if (idx !== -1) {
              const next = [...prev];
              next[idx] = routerMessage;
              return next;
            }
            return [...prev, routerMessage];
          });
        }

        if (node === "create_research_plan" && Array.isArray(update.steps)) {
          const questions = update.steps
            .map((step: unknown, index: number) => {
              if (typeof step !== "string") {
                return null;
              }
              const trimmed = step.trim();
              if (!trimmed) return null;
              return {
                step: index + 1,
                question: trimmed,
              };
            })
            .filter(Boolean);

          if (questions.length) {
            const messageId =
              streamState.generatingQuestionsMessageId ?? uuidv4();
            streamState.generatingQuestionsMessageId = messageId;

            const generatingMessage: Message = {
              type: "ai",
              id: messageId,
              content: "",
              tool_calls: [
                {
                  name: "generating_questions",
                  args: { questions },
                },
              ],
            };

            setMessages((prev) => {
              const idx = prev.findIndex((msg) => msg.id === messageId);
              if (idx !== -1) {
                const next = [...prev];
                next[idx] = generatingMessage;
                return next;
              }
              return [...prev, generatingMessage];
            });
          }
        }

        if (node === "conduct_research") {
          if (Array.isArray(update.documents)) {
            streamState.latestDocuments = update.documents;
          }

          if (streamState.generatingQuestionsMessageId) {
            const { question, queries, documents } = update;
            if (question || queries || documents) {
              setMessages((prev) => {
                const idx = prev.findIndex(
                  (msg) => msg.id === streamState.generatingQuestionsMessageId,
                );
                if (idx === -1) return prev;

                const existing = prev[idx];
                if (existing.type !== "ai") {
                  return prev;
                }

                const existingToolCall = existing.tool_calls?.[0];
                if (!existingToolCall || !existingToolCall.args?.questions) {
                  return prev;
                }

                const updatedQuestions = existingToolCall.args.questions.map(
                  (q: any) => {
                    if (q.question === question) {
                      return {
                        ...q,
                        queries: queries ?? q.queries,
                        documents: documents ?? q.documents,
                      };
                    }
                    return q;
                  },
                );

                const nextMessage: Message = {
                  ...existing,
                  tool_calls: [
                    {
                      ...existingToolCall,
                      args: {
                        ...existingToolCall.args,
                        questions: updatedQuestions,
                      },
                    },
                  ],
                };

                const next = [...prev];
                next[idx] = nextMessage;
                return next;
              });
            }
          }
        }

        if (node === "respond") {
          if (!streamState.answerHeaderInserted) {
            const answerHeaderToolMsg: Message = {
              type: "ai",
              content: "",
              tool_calls: [
                {
                  name: "answer_header",
                  args: {},
                },
              ],
            };
            setMessages((prev) => [...prev, answerHeaderToolMsg]);
            streamState.answerHeaderInserted = true;
          }

          const documents = Array.isArray(update.documents)
            ? update.documents
            : streamState.latestDocuments;

          if (documents?.length && !streamState.selectedDocumentsMessageId) {
            const messageId = uuidv4();
            streamState.selectedDocumentsMessageId = messageId;
            const selectedDocumentsMessage: Message = {
              type: "ai",
              id: messageId,
              content: "",
              tool_calls: [
                {
                  name: "selected_documents",
                  args: {
                    documents,
                  },
                },
              ],
            };

            setMessages((prev) => [...prev, selectedDocumentsMessage]);
          }
        }
      }
    },
    [updateProgress],
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
    onUpdateEvent: handleUpdateEvent,
    onFinish: async (state, meta) => {
      setIsStreaming(false);
      const currentState = streamStateRef.current;
      streamStateRef.current = null;

      const finalRunId =
        meta?.run_id ?? currentState?.runId ?? runIdRef.current;
      if (finalRunId) {
        setRunId(finalRunId);
      }

      const stateValues = state?.values as Record<string, any> | undefined;
      const outputMessages = Array.isArray(stateValues?.messages)
        ? (stateValues?.messages as Message[])
        : [];
      const documents = Array.isArray(stateValues?.documents)
        ? (stateValues?.documents as Record<string, any>[])
        : [];

      if (outputMessages.length && documents.length) {
        const lastMessage = outputMessages[outputMessages.length - 1];
        if (lastMessage?.id) {
          setMessages((prev) => {
            const idx = prev.findIndex((msg) => msg.id === lastMessage.id);
            if (idx === -1) return prev;
            const updated: Message = {
              ...prev[idx],
              content: addDocumentLinks(
                flattenMessageContent(lastMessage.content),
                documents,
              ),
            };
            const next = [...prev];
            next[idx] = updated;
            return next;
          });
        }
      }

      if (finalRunId) {
        try {
          const sharedRunURL = await shareRun(finalRunId);
          if (sharedRunURL) {
            setMessages((prevMessages) => {
              const langSmithToolCallMessage: Message = {
                type: "ai",
                id: uuidv4(),
                content: "",
                tool_calls: [
                  {
                    name: "langsmith_tool_ui",
                    args: { sharedRunURL },
                    id: sharedRunURL
                      ?.split("https://smith.langchain.com/public/")[1]
                      ?.split("/")?.[0],
                  },
                ],
              };
              return [...prevMessages, langSmithToolCallMessage];
            });
          }
        } catch (error) {
          console.error("Failed to share run", error);
        }
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

  useEffect(() => {
    const streamMessages = thread.messages ?? [];
    if (!streamMessages.length) return;

    setMessages((prev) => {
      const next = [...prev];
      const indexById = new Map(next.map((msg, idx) => [msg.id, idx]));

      for (const sdkMessage of streamMessages) {
        if (sdkMessage.type === "human") {
          continue;
        }

        const existingIndex = sdkMessage.id
          ? indexById.get(sdkMessage.id)
          : undefined;
        if (existingIndex != null) {
          next[existingIndex] = sdkMessage;
          continue;
        }

        if (sdkMessage.type === "ai") {
          let answerHeaderIndex = -1;
          for (let i = next.length - 1; i >= 0; i -= 1) {
            if (isToolMessageWithName(next[i], "answer_header")) {
              answerHeaderIndex = i;
              break;
            }
          }
          if (answerHeaderIndex !== -1) {
            next.splice(answerHeaderIndex + 1, 0, sdkMessage);
            continue;
          }
        }

        next.push(sdkMessage);
      }

      return next;
    });
  }, [thread.messages]);

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
            streamMode: ["messages", "updates"],
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

      const baseMessages = Array.isArray(threadValues.messages)
        ? (threadValues.messages as Message[])
        : [];

      const actualMessages = baseMessages.flatMap((msg, index, array) => {
        if (msg.type === "human") {
          const progressMessage: Message = {
            type: "ai",
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
          };
          return [msg, progressMessage];
        }

        if (msg.type === "ai") {
          const isLastAiMessage =
            index === array.length - 1 || array[index + 1].type === "human";
          if (isLastAiMessage) {
            const syntheticMessages: Message[] = [];
            if (threadValues.router) {
              syntheticMessages.push({
                type: "ai",
                id: uuidv4(),
                content: "",
                tool_calls: [
                  {
                    name: "router_logic",
                    args: threadValues.router,
                  },
                ],
              });
            }
            if (
              Array.isArray(threadValues.documents) &&
              threadValues.documents.length
            ) {
              syntheticMessages.push({
                type: "ai",
                id: uuidv4(),
                content: "",
                tool_calls: [
                  {
                    name: "selected_documents",
                    args: {
                      documents: threadValues.documents,
                    },
                  },
                ],
              });
            }
            syntheticMessages.push({
              type: "ai",
              content: "",
              tool_calls: [
                {
                  name: "answer_header",
                  args: {},
                },
              ],
            });
            return [...syntheticMessages, msg];
          }
          return [msg];
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
