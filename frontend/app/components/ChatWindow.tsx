"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  InputGroup,
  InputRightElement,
  Spinner,
} from "@chakra-ui/react";
import { ArrowUpIcon } from "@chakra-ui/icons";
import { Select, Link } from "@chakra-ui/react";
import { Client } from "@langchain/langgraph-sdk";

import { EmptyState } from "./EmptyState";
import { ChatMessageBubble } from "./ChatMessageBubble";
import { AutoResizeTextarea } from "./AutoResizeTextarea";
import { Message } from "../types";
import { ChatList } from "./ChatList";
import { useThread } from "../hooks/useThread";
import { useThreadList } from "../hooks/useThreadList";
import { useThreadMessages } from "../hooks/useThreadMessages";
import { useLangGraphClient } from "../hooks/useLangGraphClient";

const MODEL_TYPES = [
  "openai_gpt_3_5_turbo",
  "anthropic_claude_3_haiku",
  "google_gemini_pro",
  "fireworks_mixtral",
  "cohere_command",
];

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

export function mergeMessagesById(
  left: Message[] | Record<string, any> | null | undefined,
  right: Message[] | Record<string, any> | null | undefined,
): Message[] {
  const leftMsgs = Array.isArray(left) ? left : left?.messages;
  const rightMsgs = Array.isArray(right) ? right : right?.messages;

  const merged = (leftMsgs ?? [])?.slice();
  for (const msg of rightMsgs ?? []) {
    const foundIdx = merged.findIndex((m: any) => m.id === msg.id);
    if (foundIdx === -1) {
      merged.push(msg);
    } else {
      merged[foundIdx] = msg;
    }
  }
  return merged;
}

export function ChatWindow() {
  const router = useRouter()
  const searchParams = useSearchParams();
  const { currentThread } = useThread()
  const { threads, createThread, updateThread, deleteThread } = useThreadList();
  const { updateMessages, messages, setMessages } = useThreadMessages(currentThread?.thread_id ?? null);
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [llm, setLlm] = useState(
    searchParams.get("llm") ?? "openai_gpt_3_5_turbo",
  );
  const [llmIsLoading, setLlmIsLoading] = useState(true);
  const [assistantId, setAssistantId] = useState<string>("");

  const client = useLangGraphClient()

  const setLanggraphInfo = async () => {
    const assistantId = await getAssistantId(client);
    setAssistantId(assistantId);
  };

  useEffect(() => {
    setLlm(searchParams.get("llm") ?? defaultLlmValue);
    setLanggraphInfo();
    setLlmIsLoading(false);
  }, []);

  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    if (currentThread == null) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    const formattedMessage: Message = {
      id: Math.random().toString(),
      content: messageValue,
      type: "human",
    };
    setMessages((prevMessages) => [...prevMessages, formattedMessage]);
    setIsLoading(true);

    let renderer = new Renderer();
    renderer.paragraph = (text) => {
      return text + "\n";
    };
    renderer.list = (text) => {
      return `${text}\n\n`;
    };
    renderer.listitem = (text) => {
      return `\n• ${text}`;
    };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code,
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });
    try {
      await updateThread(currentThread.thread_id, messageValue.slice(0, 20) + "...")
      const llmDisplayName = llm ?? "openai_gpt_3_5_turbo";
      const streamResponse = client.runs.stream(currentThread["thread_id"], assistantId, {
        input: {
          messages: [formattedMessage],
        },
        config: {
          configurable: { model_name: llm },
          tags: ["model:" + llmDisplayName],
        },
        streamMode: ["messages", "values"],
      });
      await updateMessages(streamResponse);
      setIsLoading(false);
    } catch (e) {
      setIsLoading(false);
      setInput(messageValue);
      throw e;
    }
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
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
    router.push(newUrl)
  };

  const selectChat = useCallback(
    async (id: string | null) => {
      if (!id) {
        const thread = await createThread("New chat")
        insertUrlParam("threadId", thread["thread_id"])
      } else {
        insertUrlParam("threadId", id)
      }
    },
    []
  );

  return (
    <div className="flex flex-col items-center p-8 rounded grow max-h-full">
      <Flex direction={"row"} columnGap={"20px"}>
        <Flex direction={"column"} minWidth={"20%"}>
          <ChatList
            threads={threads}
            enterChat={selectChat}
            deleteChat={deleteThread}
          />
        </Flex>
        <Flex direction={"column"}>
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
              <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"white"}>
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
                <Link href="https://python.langchain.com/" color={"blue.200"}>
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
                    <option value="openai_gpt_3_5_turbo">GPT-3.5-Turbo</option>
                    <option value="anthropic_claude_3_haiku">Claude 3 Haiku</option>
                    <option value="google_gemini_pro">Google Gemini Pro</option>
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
            className="flex flex-col-reverse w-full mb-2 overflow-auto"
            ref={messageContainerRef}
          >
            {messages.length > 0 ? (
              [...messages]
                .reverse()
                .map((m, index) => (
                  <ChatMessageBubble
                    key={m.id}
                    message={{ ...m }}
                    aiEmoji="🦜"
                    isMostRecent={index === 0}
                    messageCompleted={!isLoading}
                  ></ChatMessageBubble>
                ))
            ) : (
              <EmptyState onChoice={sendInitialQuestion} />
            )}
          </div>
          <InputGroup size="md" alignItems={"center"}>
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
                icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
                type="submit"
                onClick={(e) => {
                  e.preventDefault();
                  sendMessage();
                }}
              />
            </InputRightElement>
          </InputGroup>

          {messages.length === 0 ? (
            <footer className="flex justify-center absolute bottom-8">
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
      </Flex>
    </div>
  );
}
