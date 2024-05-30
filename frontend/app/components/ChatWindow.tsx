"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "react-toastify";
import { useRouter, useSearchParams } from "next/navigation";
import "highlight.js/styles/gradient-dark.css";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  InputGroup,
  InputRightElement,
  Spinner,
  Button,
  Text,
} from "@chakra-ui/react";
import { ArrowDownIcon, ArrowUpIcon, SmallCloseIcon } from "@chakra-ui/icons";
import { Select, Link } from "@chakra-ui/react";
import { Client } from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";

import { EmptyState } from "./EmptyState";
import { ChatMessageBubble } from "./ChatMessageBubble";
import { ChatList } from "./ChatList";
import { AutoResizeTextarea } from "./AutoResizeTextarea";
import { Message } from "../types";
import { useThread } from "../hooks/useThread";
import { useThreadList } from "../hooks/useThreadList";
import { useThreadMessages } from "../hooks/useThreadMessages";
import { useLangGraphClient } from "../hooks/useLangGraphClient";
import { useStreamState } from "../hooks/useStreamState";
import { useLocalStorage } from "../hooks/useLocalStorage";

const MODEL_TYPES = ["openai_gpt_3_5_turbo", "anthropic_claude_3_haiku"];

const defaultLlmValue =
  MODEL_TYPES[Math.floor(Math.random() * MODEL_TYPES.length)];

const getAssistantId = async (client: Client) => {
  const response = await client.assistants.search({
    metadata: null,
    offset: 0,
    limit: 10,
  });
  if (response.length !== 1) {
    throw Error(
      `Expected exactly one assistant, got ${response.length} instead`,
    );
  }
  return response[0]["assistant_id"];
};

export function ChatWindow() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [llm, setLlm] = useState(
    searchParams.get("llm") ?? "openai_gpt_3_5_turbo",
  );
  const [llmIsLoading, setLlmIsLoading] = useState(true);
  const [assistantId, setAssistantId] = useState<string>("");
  const [userId, setUserId] = useLocalStorage("userId", null);

  const client = useLangGraphClient();

  const { currentThread } = useThread(userId);
  const {
    threads,
    createThread,
    updateThread,
    deleteThread,
    loadMoreThreads,
    areThreadsLoading,
  } = useThreadList(userId);
  const { streamStates, startStream, stopStream } = useStreamState();
  const streamState =
    currentThread == null
      ? null
      : streamStates[currentThread.thread_id] ?? null;
  const { refreshMessages, messages, setMessages, next, areMessagesLoading } =
    useThreadMessages(
      currentThread?.thread_id ?? null,
      streamState,
      stopStream,
    );

  const setLanggraphInfo = async () => {
    try {
      const assistantId = await getAssistantId(client);
      setAssistantId(assistantId);
    } catch (e) {
      toast.error("Could not load AI agent");
    }
  };

  const setUserInfo = () => {
    if (userId == null) {
      const userId = uuidv4();
      setUserId(userId);
    }
  };

  useEffect(() => {
    setLlm(searchParams.get("llm") ?? defaultLlmValue);
    setUserInfo();
    setLanggraphInfo();
    setLlmIsLoading(false);
  }, []);

  const config = {
    configurable: { model_name: llm },
    tags: ["model:" + llm],
  };

  const getThreadName = (messageValue: string) =>
    messageValue.length > 20 ? messageValue.slice(0, 20) + "..." : messageValue;

  const renameThread = async (messageValue: string) => {
    // NOTE: we're only setting this on the first message
    if (currentThread == null || messages.length > 1) {
      return;
    }
    const threadName = getThreadName(messageValue);
    await updateThread(currentThread["thread_id"], threadName);
  };

  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }

    const messageValue = message ?? input;
    if (messageValue === "") return;

    let thread = currentThread;
    if (thread == null) {
      const threadName = getThreadName(messageValue);
      thread = await createThread(threadName);
      insertUrlParam("threadId", thread["thread_id"]);
    }

    setInput("");
    const formattedMessage: Message = {
      id: Math.random().toString(),
      content: messageValue,
      type: "human",
    };
    setMessages((prevMessages) => [...prevMessages, formattedMessage]);
    setIsLoading(true);

    try {
      await renameThread(messageValue);
      await startStream(
        [formattedMessage],
        thread["thread_id"],
        assistantId,
        config,
      );
      await refreshMessages();
      setIsLoading(false);
    } catch (e) {
      setIsLoading(false);
      if (!(e instanceof DOMException && e.name == "AbortError")) {
        // we don't raise on "abort" signal errors
        throw e;
      }
    }
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const continueStream = async (threadId: string) => {
    try {
      setIsLoading(true);
      await startStream(null, threadId, assistantId, config);
      setIsLoading(false);
    } catch (e) {
      setIsLoading(false);
      if (!(e instanceof DOMException && e.name == "AbortError")) {
        // we don't raise on "abort" signal errors
        throw e;
      }
    }
  };

  const insertUrlParam = (key: string, value?: string) => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set(key, value ?? "");
    const newUrl =
      window.location.protocol +
      "//" +
      window.location.host +
      window.location.pathname +
      "?" +
      searchParams.toString();
    router.push(newUrl);
  };

  const selectThread = useCallback(
    async (id: string | null) => {
      if (!id) {
        const thread = await createThread("New chat");
        insertUrlParam("threadId", thread["thread_id"]);
      } else {
        insertUrlParam("threadId", id);
      }
    },
    [setMessages, createThread, insertUrlParam],
  );

  const deleteThreadAndReset = async (id: string) => {
    await deleteThread(id);
    // reset
    router.push(
      window.location.protocol +
        "//" +
        window.location.host +
        window.location.pathname,
    );
  };
  return (
    <>
      <div className="flex items-center rounded grow max-h-full">
        <Flex
          direction={"column"}
          minWidth={"212px"}
          paddingTop={"36px"}
          height={"100%"}
          marginX={"24px"}
        >
          <ChatList
            userId={userId}
            threads={threads}
            enterChat={selectThread}
            deleteChat={deleteThreadAndReset}
            areThreadsLoading={areThreadsLoading}
            loadMoreThreads={loadMoreThreads}
          />
        </Flex>
        <Flex
          direction={"column"}
          justifyContent={"center"}
          flexGrow={"1"}
          marginX={"12px"}
          height={"100%"}
        >
          {areMessagesLoading ? (
            <Spinner className="my-2" />
          ) : (
            <Flex
              direction={"column"}
              alignItems={"center"}
              flexGrow={"1"}
              margin={"24px"}
              height={"100%"}
            >
              <Flex
                direction={"column"}
                alignItems={"center"}
                marginTop={messages.length > 0 ? "" : "64px"}
              >
                <Heading
                  fontSize={messages.length > 0 ? "2xl" : "3xl"}
                  fontWeight={"medium"}
                  mb={1}
                  color={"white"}
                >
                  Chat LangChain 🦜🔗
                </Heading>
                {messages.length > 0 ? (
                  <Heading
                    fontSize="md"
                    fontWeight={"normal"}
                    mb={1}
                    color={"white"}
                  >
                    We appreciate feedback!
                  </Heading>
                ) : (
                  <Heading
                    fontSize="xl"
                    fontWeight={"normal"}
                    color={"white"}
                    marginTop={"10px"}
                    textAlign={"center"}
                  >
                    Ask me anything about LangChain&apos;s{" "}
                    <Link
                      href="https://python.langchain.com/"
                      color={"blue.200"}
                    >
                      Python documentation!
                    </Link>
                  </Heading>
                )}
                <div className="text-white flex flex-wrap items-center mt-4">
                  <div className="flex items-center mb-2">
                    <span className="shrink-0 mr-2">Powered by</span>
                    {llmIsLoading ? (
                      <Spinner className="my-2"></Spinner>
                    ) : (
                      <Select
                        value={llm}
                        onChange={(e) => {
                          insertUrlParam("llm", e.target.value);
                          setLlm(e.target.value);
                        }}
                        width={"240px"}
                      >
                        <option value="openai_gpt_3_5_turbo">
                          GPT-3.5-Turbo
                        </option>
                        <option value="anthropic_claude_3_haiku">
                          Claude 3 Haiku
                        </option>
                        <option value="google_gemini_pro">
                          Google Gemini Pro
                        </option>
                        <option value="fireworks_mixtral">
                          Mixtral (via Fireworks.ai)
                        </option>
                        <option value="cohere_command">Cohere</option>
                      </Select>
                    )}
                  </div>
                </div>
              </Flex>
              <div
                className="flex flex-col-reverse w-full mb-2 overflow-auto max-h-[75vh]"
                ref={messageContainerRef}
              >
                {messages.length > 0 && currentThread != null ? (
                  <>
                    {next.length > 0 &&
                      streamStates[currentThread.thread_id]?.status !==
                        "inflight" && (
                        <Button
                          key={"continue-button"}
                          backgroundColor={"rgb(58, 58, 61)"}
                          _hover={{ backgroundColor: "rgb(78,78,81)" }}
                          onClick={() =>
                            continueStream(currentThread["thread_id"])
                          }
                        >
                          <ArrowDownIcon color={"white"} marginRight={"4px"} />
                          <Text color={"white"}>Click to continue</Text>
                        </Button>
                      )}
                    {[...messages].reverse().map((m, index) => (
                      <ChatMessageBubble
                        key={m.id}
                        message={{ ...m }}
                        feedbackUrls={
                          streamStates[currentThread.thread_id]?.feedbackUrls
                        }
                        aiEmoji="🦜"
                        isMostRecent={index === 0}
                        messageCompleted={!isLoading}
                      />
                    ))}
                  </>
                ) : (
                  <EmptyState onChoice={sendInitialQuestion} />
                )}
              </div>
              <InputGroup
                size="md"
                alignItems={"center"}
                maxWidth={messages.length === 0 ? "1024px" : "100%"}
                marginY={"12px"}
              >
                <AutoResizeTextarea
                  value={input}
                  maxRows={5}
                  marginRight={"56px"}
                  placeholder="What does RunnablePassthrough.assign() do?"
                  textColor={"white"}
                  borderColor={"rgb(58, 58, 61)"}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    } else if (e.key === "Enter" && e.shiftKey) {
                      e.preventDefault();
                      setInput(input + "\n");
                    }
                  }}
                />
                <InputRightElement h="full">
                  <IconButton
                    colorScheme="blue"
                    rounded={"full"}
                    aria-label="Send"
                    icon={isLoading ? <SmallCloseIcon /> : <ArrowUpIcon />}
                    type="submit"
                    onClick={(e) => {
                      e.preventDefault();
                      if (currentThread == null) {
                        return;
                      }

                      if (isLoading) {
                        stopStream?.(currentThread.thread_id);
                      } else {
                        sendMessage();
                      }
                    }}
                  />
                </InputRightElement>
              </InputGroup>
              {messages.length === 0 ? (
                <footer className="flex justify-center mt-auto h-4 fixed bottom-4">
                  <a
                    href="https://github.com/langchain-ai/chat-langchain"
                    target="_blank"
                    className="text-white flex items-center"
                  >
                    <img src="/images/github-mark.svg" className="h-4 mr-1" />
                    <span>View Source</span>
                  </a>
                </footer>
              ) : (
                ""
              )}
            </Flex>
          )}
        </Flex>
      </div>
    </>
  );
}
