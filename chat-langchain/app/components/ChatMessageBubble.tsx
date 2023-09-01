import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { emojisplosion } from "emojisplosion";
import { useState } from "react";

export type Message = {
  id: string;
  createdAt?: Date | undefined;
  content: string;
  role: 'system' | 'user' | 'assistant' | 'function';
  name?: string | undefined;
  function_call?: { name: string };
};

export function ChatMessageBubble(props: {
  message: Message;
  aiEmoji?: string;
  feedback: number | null;
  sendFeedback: (feedback: 0 | 1) => void;
  isMostRecent: boolean;
  messageCompleted: boolean;
}) {
  const colorClassName =
    props.message.role === "user" ? "bg-sky-600" : "bg-slate-50 text-black";
  const alignmentClassName =
    props.message.role === "user" ? "ml-auto" : "mr-auto";
  const prefix = props.message.role === "user" ? "🧑" : props.aiEmoji;

  const [feedbackColor, setFeedbackColor] = useState("");

  const cumulativeOffset = function(element: HTMLElement | null) {
      var top = 0, left = 0;
      do {
          top += element?.offsetTop  || 0;
          left += element?.offsetLeft || 0;
          element = (element?.offsetParent as HTMLElement) || null;
      } while(element);

      return {
          top: top,
          left: left
      };
  };

  const animateButton = (buttonId: string) => {
    const button = document.getElementById(buttonId);
    button!.classList.add("animate-ping");
    setTimeout(() => {
      button!.classList.remove("animate-ping");
    }, 500);

    emojisplosion({
      emojiCount: 10,
      uniqueness: 1,
      position() {
        const offset = cumulativeOffset(button);

        return {
          x: offset.left + button!.clientWidth / 2,
          y: offset.top + button!.clientHeight / 2,
        };
      },
      emojis: buttonId === "upButton" ? ["👍"] : ["👎"],
    });
  };

  return (
    <div className="mt-4 flex flex-col">
      <div
        className={`${alignmentClassName} ${colorClassName} ${feedbackColor} rounded px-4 py-2 max-w-[80%] mb-1 flex break-words`}
      >
        <div className="mr-2">{prefix}</div>
        <div
          className="whitespace-pre-wrap"
          dangerouslySetInnerHTML={{ __html: props.message.content }}
        ></div>
      </div>
      {props.message.role !== "user" && props.isMostRecent && props.messageCompleted && (
        <div className="relative flex space-x-1 items-start justify-start">
          <button
            className={`text-sm rounded ${props.feedback === null ? "hover:bg-green-200" : ""}`}
            id="upButton"
            type="button"
            onClick={() => {
              if (props.feedback === null) {
                props.sendFeedback(1);
                animateButton("upButton")
                setFeedbackColor("border-4 border-green-300");
              } else {
                toast.error("You have already provided your feedback.");
              }
            }}
          >
            👍
          </button>
          <button
            className={`text-sm rounded ${props.feedback === null ? "hover:bg-red-200" : ""}`}
            id="downButton"
            type="button"
            onClick={() => {
              if (props.feedback === null) {
                props.sendFeedback(0);
                animateButton("downButton")
                setFeedbackColor("border-4 border-red-300")
              } else {
                toast.error("You have already provided your feedback.");
              }
            }}
          >
            👎
          </button>
        </div>
      )}
    </div>
  );
}
