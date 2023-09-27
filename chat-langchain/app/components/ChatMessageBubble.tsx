import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { emojisplosion } from "emojisplosion";
import { FormEvent, useState } from "react";
import { SourceBubble, Source } from "./SourceBubble";
import { Flex, Box, Heading, HStack, VStack, Divider } from "@chakra-ui/react";
import { InlineCitation } from "./InlineCitation";
import { v4 as uuidv4 } from "uuid";

export type Message = {
  id: string;
  createdAt?: Date;
  content: string;
  role: "system" | "user" | "assistant" | "function";
  runId?: string;
  sources?: Source[];
  name?: string;
  function_call?: { name: string };
};
export interface Feedback {
  feedback_id: string;
  run_id: string;
  key: string;
  score: number;
  comment?: string;
}

export function ChatMessageBubble(props: {
  message: Message;
  aiEmoji?: string;
  isMostRecent: boolean;
  messageCompleted: boolean;
  apiBaseUrl: string;
}) {
  const { role, content, runId } = props.message;
  const isUser = role === "user";

  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [comment, setComment] = useState("");
  const [showCommentForm, setShowCommentForm] = useState(false);
  const [feedbackColor, setFeedbackColor] = useState("");

  const cumulativeOffset = function (element: HTMLElement | null) {
    var top = 0,
      left = 0;
    do {
      top += element?.offsetTop || 0;
      left += element?.offsetLeft || 0;
      element = (element?.offsetParent as HTMLElement) || null;
    } while (element);

    return {
      top: top,
      left: left,
    };
  };

  async function handleCommentSubmission(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const score = typeof feedback?.score === "number" ? feedback.score : 0;
    await sendFeedback(score, "user_score");
  }

  const sendFeedback = async (score: number, key: string) => {
    let run_id = runId;
    console.log("Sending feedback", run_id, score, key, comment);
    if (run_id === undefined) {
      return;
    }
    if (isLoading) {
      return;
    }
    setIsLoading(true);
    console.log("Still loading/sending feedback", run_id, score, key, comment);
    let apiBaseUrl = props.apiBaseUrl;
    let feedback_id = feedback?.feedback_id ?? uuidv4();
    try {
      const response = await fetch(apiBaseUrl + "/feedback", {
        method: feedback?.feedback_id ? "PATCH" : "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          score,
          run_id,
          key,
          feedback_id,
          comment,
        }),
      });
      console.log("Response", response);
      const data = await response.json();
      if (data.code === 200) {
        setFeedback({ run_id, score, key, feedback_id });
        score == 1 ? animateButton("upButton") : animateButton("downButton");
        if (comment) {
          setComment("");
          setShowCommentForm(false);
        } else {
          setShowCommentForm(true);
        }
      }
    } catch (e: any) {
      console.error("Error:", e);
      toast.error(e.message);
    }
    setIsLoading(false);
  };

  const sources = props.message.sources ?? [];

  // Use an array of highlighted states as a state since React
  // complains when creating states in a loop
  const [highlighedSourceLinkStates, setHighlightedSourceLinkStates] = useState(
    sources.map(() => false)
  );
  const matches = Array.from(content.matchAll(/\[(\d+)\]/g));
  const answerElements = [];

  let previousSliceIndex = 0;
  for (const match of matches) {
    const sourceNumber = parseInt(match[1], 10);
    if (match.index && sources[sourceNumber]) {
      answerElements.push(
        <span
          key={`content:${previousSliceIndex}`}
          dangerouslySetInnerHTML={{
            __html: content.slice(previousSliceIndex, match.index),
          }}
        ></span>
      );
      answerElements.push(
        <InlineCitation
          key={`citation:${previousSliceIndex}`}
          source={sources[sourceNumber]}
          sourceNumber={sourceNumber}
          highlighted={highlighedSourceLinkStates[sourceNumber]}
          onMouseEnter={() =>
            setHighlightedSourceLinkStates(
              sources.map((_, i) => i === sourceNumber)
            )
          }
          onMouseLeave={() =>
            setHighlightedSourceLinkStates(sources.map(() => false))
          }
        ></InlineCitation>
      );
      previousSliceIndex = match.index + match[0].length;
    }
  }
  answerElements.push(
    <span
      key={`content:${previousSliceIndex}`}
      dangerouslySetInnerHTML={{
        __html: content.slice(previousSliceIndex),
      }}
    ></span>
  );

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
    <VStack align={"start"} paddingBottom={"20px"}>
      {!isUser && sources.length > 0 && (
        <>
          <Flex direction={"column"} width={"100%"}>
            <VStack spacing={"5px"} align={"start"} width={"100%"}>
              <Heading
                fontSize="lg"
                fontWeight={"medium"}
                mb={1}
                color={"blue.300"}
                paddingBottom={"10px"}
              >
                Sources
              </Heading>
              <HStack spacing={"10px"} maxWidth={"100%"}>
                {sources.map((source, index) => (
                  <Box key={index} alignSelf={"stretch"} width={40}>
                    <SourceBubble
                      source={source}
                      highlighted={highlighedSourceLinkStates[index]}
                      onMouseEnter={() =>
                        setHighlightedSourceLinkStates(
                          sources.map((_, i) => i === index)
                        )
                      }
                      onMouseLeave={() =>
                        setHighlightedSourceLinkStates(sources.map(() => false))
                      }
                    />
                  </Box>
                ))}
              </HStack>
            </VStack>
          </Flex>

          <Heading
            fontSize="lg"
            fontWeight={"medium"}
            mb={1}
            color={"blue.300"}
            paddingTop={"20px"}
          >
            Answer
          </Heading>
        </>
      )}
      {isUser ? (
        <Heading size={"lg"} fontWeight={"medium"} color={"white"}>
          {content}
        </Heading>
      ) : (
        <div className="whitespace-pre-wrap" style={{ color: "white" }}>
          {answerElements}
        </div>
      )}
      {props.message.role !== "user" &&
        props.isMostRecent &&
        props.messageCompleted && (
          <div className="relative flex space-x-1 items-start justify-start">
            <button
              className={`text-sm rounded ${
                feedback === null ? "hover:bg-green-200" : ""
              }`}
              id="upButton"
              type="button"
              onClick={() => {
                if (feedback === null && props.message.runId) {
                  sendFeedback(1, "user_score");
                  animateButton("upButton");
                  setFeedbackColor("border-4 border-green-300");
                } else {
                  toast.error("You have already provided your feedback.");
                }
              }}
            >
              👍
            </button>
            <button
              className={`text-sm rounded ${
                feedback === null ? "hover:bg-red-200" : ""
              }`}
              id="downButton"
              type="button"
              onClick={() => {
                if (feedback === null && props.message.runId) {
                  sendFeedback(0, "user_score");
                  animateButton("downButton");
                  setFeedbackColor("border-4 border-red-300");
                } else {
                  toast.error("You have already provided your feedback.");
                }
              }}
            >
              👎
            </button>
            <div
              className={`${
                // Synchronous feedback currently flakey. Need to wait for
                // updates to the queue.
                false && feedback && showCommentForm ? "" : "hidden"
              } min-w-[480px]`}
            >
              <form onSubmit={handleCommentSubmission} className="relative">
                <input
                  className="mr-8 p-4 rounded w-full border mt-2"
                  value={comment}
                  placeholder={
                    feedback?.score === 1
                      ? "Anything else you'd like to add about this response?"
                      : "What would the correct or preferred response have been?"
                  }
                  onChange={(e) => setComment(e.target.value)}
                />
                <div
                  role="status"
                  className={`${
                    isLoading ? "" : "hidden"
                  } flex justify-center absolute top-[24px] right-[16px]`}
                >
                  <svg
                    aria-hidden="true"
                    className="w-6 h-6 text-slate-200 animate-spin dark:text-slate-200 fill-sky-800"
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
              </form>
            </div>
          </div>
        )}

      {!isUser && <Divider marginTop={"20px"} marginBottom={"20px"} />}
    </VStack>
  );
}
