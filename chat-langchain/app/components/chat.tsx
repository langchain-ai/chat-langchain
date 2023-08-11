"use client";
import React, { useEffect, useRef, useState } from "react";

const Chat = () => {
  const [messages, setMessages] = useState<
    Array<{ sender: string; message: string; isCode: boolean }>
  >([]);
  const [model, setModel] = useState("openai");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [message, setMessage] = useState(""); // Add this line

  const sendMessage = (event: { preventDefault: () => void }) => {
    event.preventDefault();
    if (message === "") {
      return;
    }
    let isCode = false;

    fetch("http://0.0.0.0:8080/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        model: model,
      }),
    })
      .then((response) => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        let accumulatedMessage = "";
        let prevIsCode = null;

        reader
          .read()
          .then(function processText({
            done,
            value,
          }: {
            done: any;
            value: any;
          }) {
            if (done) {
              console.log("Stream complete");
              return;
            }

            let result = decoder.decode(value);
            if (result.includes("``")) {
              isCode = !isCode;
            } else {
              result = result.replace(/`/g, "");
              accumulatedMessage += result;
            }

            setMessages((prevMessages) => {
              let newMessages = [...prevMessages];
              if (isCode !== prevIsCode) {
                newMessages.push({
                  sender: isCode ? "" : "Bot",
                  message: accumulatedMessage.trim(),
                  isCode: isCode,
                });
                if (newMessages.length > 1) {
                  accumulatedMessage = "";
                }
              } else if (newMessages.length > 0) {
                newMessages[newMessages.length - 1] = {
                  sender: isCode ? "" : "Bot",
                  message: accumulatedMessage.trim(),
                  isCode: isCode,
                };
              }
              prevIsCode = isCode;
              return newMessages;
            });
            setMessage("");
            return reader.read().then(processText);
          });
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messagesEndRef]);

  return (
    <div className="rounded-2xl border-zinc-100  lg:border lg:p-6 bg-slate-200">
      <div className="w-full mb-5 rounded-lg px-4 py-5 shadow-lg ring-1 ring-zinc-100 sm:px-6 bg-slate-100">
        <h4 className="text-center text-3xl font-sans font-medium">
          Chat Langchain
        </h4>
        <span className="mx-auto flex font-mono flex-grow text-gray-400 clear-both mb-5 mt-1 text-sm justify-center">
          A chatbot for everything Langchain.
        </span>
        <hr className="border-gray-500 my-5" />
        <div
          id="messages"
          className="overflow-auto m-8"
          style={{ maxHeight: "500px" }}
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
                  msg.sender === "You" ? "client-message" : "server-message"
                }
              >
                {msg.sender && <strong>{msg.sender}: </strong>}
                {msg.isCode ? (
                  <code
                    style={{
                      color: "white",
                      backgroundColor: "rgba(0, 0, 0, 0.8)",
                      opacity: "0.9",
                      padding: "4px",
                      margin: "4px",
                      borderRadius: "5px",
                      overflow: "auto",
                      boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.5))",
                      transition: "ease-in-out 0.15s",
                    }}
                    dangerouslySetInnerHTML={{ __html: msg.message }}
                  ></code>
                ) : (
                  msg.message
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <form className="flex justify-between mt-5" onSubmit={sendMessage}>
          <input
            type="text"
            className="w-4/5 bg-gray-800 text-white rounded p-2 mr-2"
            placeholder="Write your question"
            id="messageText"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
          <select
            id="modelType"
            className="w-1/5 bg-gray-800 text-white rounded p-2 mr-2"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
          </select>
          <button
            id="send"
            type="submit"
            className="bg-blue-500 text-white rounded p-2"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default Chat;
