"use client";

import React, { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

const Chat = () => {
  const [messages, setMessages] = useState<
    Array<{ sender: string; message: string }>
  >([]);
  const [model, setModel] = useState("openai");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [message, setMessage] = useState("");
  const [hasInteracted, setHasInteracted] = useState(false);

  const sendMessage = (event: { preventDefault: () => void }) => {
    event.preventDefault();
    if (message === "") {
      return;
    }
    setFeedback(null);
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "You", message },
    ]);

    fetch("https://chat-langchain.fly.dev/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        model: model,
      }),
    }).then((response) => {
      if (!response.body) {
        throw new Error("Response body is null");
      }
      setMessage("");
      setHasInteracted(true);
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let accumulatedMessage = "";
      let messageIndex: number | null = null;

      let renderer = new Renderer();
      renderer.code = function (code, language) {
        const validLanguage = hljs.getLanguage(language || "")
          ? language
          : "plaintext";
        const highlightedCode = hljs.highlight(
          validLanguage || "plaintext",
          code
        ).value;
        return `<pre class="highlight bg-gray-700" style="padding: 10px; margin: 10px; border-radius: 5px; overflow: auto;"><code class="${language}" style="color: #d6e2ef;">${highlightedCode}</code></pre>`;
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
                sender: "Bot",
                message: parsedResult.trim(),
              });
            } else {
              newMessages[messageIndex].message = parsedResult.trim();
            }
            return newMessages;
          });
          return reader.read().then(processText);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    });
  };

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const [feedback, setFeedback] = useState<number | null>(null);

  const sendFeedback = (score: number | null) => {
    if (feedback !== null) return;

    setFeedback(score);
    fetch("https://chat-langchain.fly.dev/feedback", {
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
    fetch("https://chat-langchain.fly.dev/get_trace", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.text()) // change this to handle text response
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
    <div className="rounded-2xl border-zinc-100 lg:border lg:p-6 bg-slate-200 w-full sm:w-[60%] lg:w-[75%]">
      <div className="w-full mb-5 rounded-lg px-4 py-5 shadow-lg ring-1 ring-zinc-100 sm:px-6 bg-slate-100">
        <h4 className="text-center text-3xl font-sans font-bold pb-2 text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-teal-800">
          Chat LangChain
        </h4>
        <span className="mx-auto flex font-light flex-grow text-gray-400 clear-both text-sm justify-center font-mono">
          A chatbot for everything LangChain.
        </span>
        <hr className="border-gray-500 my-4" />
        <div className="flex flex-col">
          <div>
            <button
              type="button"
              className="text-black text-xs border rounded p-1 float-right"
              onClick={() => viewTrace()}
            >
              🛠️ view trace
            </button>
          </div>
          <div
            className="my-2 font-sans font-normal max-h-[calc(100vh-28rem)] overflow-y-scroll"
          >
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${
                  msg.sender === "You" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={
                    msg.sender === "You"
                      ? "bg-gray-700 border border-gray-600 rounded-md p-2 m-2 text-xs sm:text-base text-white text-right max-w-[80%] inline-block"
                      : "bg-slate-200 border border-gray-200 rounded-md p-2 m-2 text-xs sm:text-base text-left max-w-[80%] inline-block"
                  }
                >
                  {msg.sender && (
                    <strong className="font-medium">{msg.sender}: </strong>
                  )}
                  <div dangerouslySetInnerHTML={{ __html: msg.message }}></div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <form
            className="flex flex-col sm:flex-row justify-between mt-5"
            onSubmit={sendMessage}
          >
            <input
              type="text"
              className="w-full sm:w-[70%] text-gray-800 rounded p-2 mr-2 focus:outline-none focus:ring-1 focus:ring-teal-600"
              placeholder="Write your question"
              id="messageText"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
            />
            <div className="flex flex-row mt-2 sm:mt-0">
              <select
                id="modelType"
                className="w-full bg-gray-800 text-white rounded p-2 mr-2"
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
                  if (feedback === null) {
                    sendFeedback(1);
                  }
                }}
                disabled={feedback !== null || !hasInteracted}
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
                  if (feedback === null) {
                    sendFeedback(0);
                  }
                }}
                disabled={feedback !== null || !hasInteracted}
                id="downButton"
              >
                👎
              </button>
              <button
                id="send"
                type="submit"
                className="w-16 sm:w-auto text-sm bg-gradient-to-r from-teal-600 to-teal-800 text-white rounded p-2"
              >
                Send
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chat;
