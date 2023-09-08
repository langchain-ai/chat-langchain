"use client";

import React, { useRef, useState, FormEventHandler, FormEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import { EmptyState } from "../components/EmptyState";
import { ChatMessageBubble } from "../components/ChatMessageBubble";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Heading, Flex, Button, IconButton, Input, InputGroup, InputRightElement, Spinner} from '@chakra-ui/react'
import { ArrowUpIcon, SpinnerIcon } from "@chakra-ui/icons";

export function ChatWindow(props: {
  apiBaseUrl: string;
  placeholder?: string;
  titleText?: string;
}) {
  const conversationId = uuidv4();
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<
    Array<{
      id: string;
      message: string;
      role: "function" | "user" | "assistant" | "system";
    }>
  >([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<number | null>(null);

  const [chatHistory, setChatHistory] = useState<
    { question: string; result: string }[]
  >([]);

  const {
    apiBaseUrl,
    placeholder,
    titleText = "An LLM",
  } = props;


  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Math.random().toString(), message: messageValue, role: "user" },
    ]);
    setFeedback(null);
    setIsLoading(true);
    let response;
    try {
      response = await fetch(apiBaseUrl + "/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: messageValue,
          history: chatHistory,
          conversation_id: conversationId,
        }),
      });
    } catch (e) {
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
      throw e;
    }
    if (!response.body) {
      throw new Error("Response body is null");
    }
    const reader = response.body.getReader();
    let decoder = new TextDecoder();

    let accumulatedMessage = "";
    let messageIndex: number | null = null;

    let renderer = new Renderer();
    renderer.paragraph = function (text) {
      return text;
    };
    renderer.code = function (code, language) {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });

    reader
      .read()
      .then(function processText(
        res: ReadableStreamReadResult<Uint8Array>
      ): Promise<void> {
        const { done, value } = res;
        if (done) {
          console.log("Stream complete");
          setChatHistory((prevChatHistory) => [
            ...prevChatHistory,
            { question: messageValue, result: accumulatedMessage },
          ]);
          return Promise.resolve();
        }

        let result = decoder.decode(value);
        accumulatedMessage += result;
        let parsedResult = marked.parse(accumulatedMessage);

        setMessages((prevMessages) => {
          let newMessages = [...prevMessages];
          if (messageIndex === null) {
            messageIndex = newMessages.length;
            newMessages.push({
              id: Math.random().toString(),
              message: parsedResult.trim(),
              role: "assistant",
            });
          } else {
            newMessages[messageIndex].message = parsedResult.trim();
          }
          return newMessages;
        });
        setIsLoading(false);
        return reader.read().then(processText);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
}

  const sendFeedback = async (score: number | null) => {
    if (feedback !== null) return;

    setFeedback(score);
    try {
      const response = await fetch(apiBaseUrl + "/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          score: score,
        }),
      });
      const data = await response.json();
      if (data.code === 200) {
        score == 1 ? animateButton("upButton") : animateButton("downButton");
      }
    } catch (e: any) {
      console.error("Error:", e);
      toast.error(e.message);
    }
  };

  const animateButton = (buttonId: string) => {
    const button = document.getElementById(buttonId);
    button!.classList.add("animate-ping");
    setTimeout(() => {
      button!.classList.remove("animate-ping");
    }, 500);
  };

  const viewTrace = async () => {
    try {
      const response = await fetch(apiBaseUrl + "/get_trace", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await response.json();

      if (data.code === 400 && data.result === "No chat session found") {
        toast.error("Unable to view trace");
        throw new Error("Unable to view trace");
      } else {
        const url = data.replace(/['"]+/g, "");
        window.open(url, "_blank");
      }
    } catch (e: any) {
      console.error("Error:", e);
      toast.error(e.message);
    }
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question)
  };

  return (
    <div
      className="flex flex-col items-center p-8 rounded grow max-h-full"
      background-color="black"
    >
      {messages.length > 0 && (
        <Flex direction={"column"} alignItems={"center"} paddingBottom={"20px"}>
          <Heading fontSize="2xl" fontWeight={"medium"} mb={1} color={"white"}>{titleText}</Heading>
          <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"white"}>
            We appreciate feedback!
          </Heading>
        </Flex>
      )}
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0
          ? [...messages]
              .reverse()
              .map((m, index) => (
                <ChatMessageBubble
                  key={m.id}
                  message={{ id: m.id, content: m.message, role: m.role }}
                  aiEmoji="🦜"
                  sendFeedback={sendFeedback}
                  feedback={feedback}
                  isMostRecent={index === 0}
                  messageCompleted={!isLoading}
                ></ChatMessageBubble>
              ))
          : <EmptyState onChoice={sendInitialQuestion} />}
      </div>

      <div className="flex w-full flex-row-reverse mb-2">
              <Button onClick={() => viewTrace()} textColor={"white"} backgroundColor={"rgb(58, 58, 61)"} _hover={{"background-color": "rgb(78,78,81)"}} size="sm">
              🛠️ view trace
              </Button>
        </div>

      <InputGroup size='md' alignItems={"center"} >
        <Input
        value={input}
          height={"55px"}
          rounded={"full"}
          type={'text'}
          placeholder='What is LangChain Expression Language?'
          textColor={"white"}
          borderColor={"rgb(58, 58, 61)"}
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage()
            }}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <InputRightElement h="full" paddingRight={"15px"}>
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              sendMessage()
              }}
          />
        </InputRightElement>
      </InputGroup>
    </div>
  );
}
