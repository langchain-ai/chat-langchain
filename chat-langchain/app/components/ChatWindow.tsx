"use client";

import React, { useRef, useState, useEffect } from "react";
import type { FormEvent } from "react";
import { ChatMessageBubble } from "../components/ChatMessageBubble";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export function ChatWindow(props: {
  endpoint: string;
  emptyStateComponent: React.ReactElement;
  placeholder?: string;
  titleText?: string;
}) {
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<
    Array<{
      id: string;
      message: string;
      role: "function" | "user" | "assistant" | "system";
    }>
  >([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("openai");
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<number | null>(null);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [chatHistory, setChatHistory] = useState<{question: string, result: string}[]>([]);

  const {
    endpoint,
    emptyStateComponent,
    placeholder,
    titleText = "An LLM",
  } = props;

  function sendMessage(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    setHasInteracted(true);
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Math.random().toString(), message: input.trim(), role: "user" },
    ]);
    setFeedback(null);
    setIsLoading(true);
    fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: input,
        model: model,
        history: chatHistory,
      }),
    }).then((response) => {
      if (!response.body) {
        throw new Error("Response body is null");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let accumulatedMessage = "";
      let messageIndex: number | null = null;

      let renderer = new Renderer();
      renderer.paragraph = function(text) {
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
        return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; word-wrap: break-word;"><code class="${language}" style="color: #d6e2ef;">${highlightedCode}</code></pre>`;
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
          setChatHistory(prevChatHistory => [...prevChatHistory, {question: input, result: accumulatedMessage}]);
          return Promise.resolve();
        }
    
        let result = decoder.decode(value);
        accumulatedMessage += result;
        let parsedResult = marked.parse(accumulatedMessage);
    
        setMessages((prevMessages) => {
            let newMessages = [...prevMessages];
            if (messageIndex === null) {
              messageIndex = newMessages.length;
              newMessages.push({ id: Math.random().toString(), message: parsedResult.trim(), role: "assistant" });
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
    });
  }

  const sendFeedback = (score: number | null) => {
    if (feedback !== null) return;

    setFeedback(score);
    fetch("http://0.0.0.0:8080/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        score: score,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.code === 200) {
          score == 1 ? animateButton("upButton") : animateButton("downButton");
        }
        console.log(data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const animateButton = (buttonId: string) => {
    const button = document.getElementById(buttonId);
    button!.classList.add("animate-ping");
    setTimeout(() => {
      button!.classList.remove("animate-ping");
    }, 500);
  };

  const viewTrace = () => {
    console.log("IN VIEW TRACE")
    fetch("http://0.0.0.0:8080/get_trace", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.text())
      .then((data) => {
        console.log(data);
        const url = data.replace(/['"]+/g, "");
        window.open(url, "_blank");
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  return (
    <div
      className={`flex flex-col items-center p-8 rounded grow max-h-[calc(100%-2rem)] ${
        messages.length > 0 ? "border" : ""
      }`}
    >
      <h2 className={`${messages.length > 0 ? "" : "hidden"} text-2xl mb-1`}>
        {titleText}
      </h2>
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0
          ? [...messages]
              .reverse()
              .map((m) => (
                <ChatMessageBubble
                  key={m.id}
                  message={{ id: m.id, content: m.message, role: m.role }}
                  aiEmoji="🦜"
                ></ChatMessageBubble>
              ))
          : emptyStateComponent}
      </div>

      <div className="flex w-full flex-row-reverse mb-2">
            <button
              type="button"
              className="text-xs border rounded p-1 float-right"
              onClick={() => viewTrace()}
            >
              🛠️ view trace
            </button>
        </div>

      <form onSubmit={sendMessage} className="flex w-full">
        <input
          className="flex-grow mr-2 p-4 rounded"
          placeholder={placeholder}
          onChange={(e) => setInput(e.target.value)}
          style={{minWidth: '50px'}}
        />
        <select
          id="modelType"
          className="bg-gray-800 text-white rounded p-2 mr-2 w-full sm:w-auto"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
        </select>
        <button
          type="button"
          className={`w-16 sm:w-auto text-white border rounded p-2 mr-1 hover:border-green-400 ${
            feedback === 1 ? "bg-green-100" : ""
          }`}
          onClick={() => {
            if (feedback === null && hasInteracted) {
              sendFeedback(1);
            } else {
              toast.error("You have already provided your feedback.");
            }
          }}
          id="upButton"
        >
          👍
        </button>
        <button
          type="button"
          className={`w-16 sm:w-auto text-white border rounded p-2 mr-2 hover:border-red-400 ${
            feedback === 0 ? "bg-red-100" : ""
          }`}
          onClick={() => {
            if (feedback === null && hasInteracted) {
              sendFeedback(0);
            } else {
              toast.error("You have already provided your feedback.");
            }
          }}
          id="downButton"
        >
          👎
        </button>
        <button
          type="submit"
          className="flex-shrink p-2 w-16 sm:w-auto bg-sky-600 rounded"
        >
          <div
            role="status"
            className={`${isLoading ? "" : "hidden"} flex justify-center`}
          >
            <svg
              aria-hidden="true"
              className="w-6 h-6 text-white animate-spin dark:text-white fill-sky-800"
              viewBox="0 0 100 101"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                fill="currentColor"
              />
              <path
                d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                fill="currentFill"
              />
            </svg>
            <span className="sr-only">Loading...</span>
          </div>
          <span className={isLoading ? "hidden" : ""}>Send</span>
        </button>
      </form>
    </div>
  );
}
