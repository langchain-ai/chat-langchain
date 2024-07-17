import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { emojisplosion } from "emojisplosion";
import { useState, useRef } from "react";
import * as DOMPurify from "dompurify";
import { Renderer, marked } from "marked";
import hljs from "highlight.js";
import {
  VStack,
  Flex,
  Heading,
  HStack,
  Box,
  Button,
  Divider,
} from "@chakra-ui/react";

import { SourceBubble } from "./SourceBubble";
import { Message, Source, Feedback } from "../types";
import { hasAllowedURLPrefix } from "../utils/sendFeedback";
import { sendFeedback } from "../utils/sendFeedback";
import { RESPONSE_FEEDBACK_KEY, SOURCE_CLICK_KEY } from "../utils/constants";
import { InlineCitation } from "./InlineCitation";

const filterSources = (sources: Source[]) => {
  const filtered: Source[] = [];
  const urlMap = new Map<string, number>();
  const indexMap = new Map<number, number>();
  sources.forEach((source, i) => {
    const { url } = source;
    const index = urlMap.get(url);
    if (index === undefined) {
      urlMap.set(url, i);
      indexMap.set(i, filtered.length);
      filtered.push(source);
    } else {
      const resolvedIndex = indexMap.get(index);
      if (resolvedIndex !== undefined) {
        indexMap.set(i, resolvedIndex);
      }
    }
  });
  return { filtered, indexMap };
};

const getMarkedRenderer = () => {
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
  return renderer;
};

const createAnswerElements = (
  content: string,
  filteredSources: Source[],
  sourceIndexMap: Map<number, number>,
  highlighedSourceLinkStates: boolean[],
  setHighlightedSourceLinkStates: React.Dispatch<
    React.SetStateAction<boolean[]>
  >,
) => {
  const matches = Array.from(content.matchAll(/\[\^?\$?{?(\d+)}?\^?\]/g));
  const elements: JSX.Element[] = [];
  let prevIndex = 0;

  const renderer = getMarkedRenderer();
  marked.setOptions({ renderer });

  matches.forEach((match) => {
    const sourceNum = parseInt(match[1], 10);
    const resolvedNum = sourceIndexMap.get(sourceNum) ?? 10;
    if (match.index !== null && resolvedNum < filteredSources.length) {
      elements.push(
        <span
          key={`content:${prevIndex}`}
          dangerouslySetInnerHTML={{
            __html: DOMPurify.sanitize(
              marked.parse(content.slice(prevIndex, match.index)).trimEnd(),
            ),
          }}
        ></span>,
      );
      elements.push(
        <InlineCitation
          key={`citation:${prevIndex}`}
          source={filteredSources[resolvedNum]}
          sourceNumber={resolvedNum}
          highlighted={highlighedSourceLinkStates[resolvedNum]}
          onMouseEnter={() =>
            setHighlightedSourceLinkStates(
              filteredSources.map((_, i) => i === resolvedNum),
            )
          }
          onMouseLeave={() =>
            setHighlightedSourceLinkStates(filteredSources.map(() => false))
          }
        />,
      );
      prevIndex = (match?.index ?? 0) + match[0].length;
    }
  });
  elements.push(
    <span
      key={`content:${prevIndex}`}
      dangerouslySetInnerHTML={{
        __html: DOMPurify.sanitize(
          marked.parse(content.slice(prevIndex)).trimEnd(),
        ),
      }}
    ></span>,
  );
  return elements;
};

export function ChatMessageBubble(props: {
  message: Message;
  feedbackUrls?: Record<string, string[]>;
  aiEmoji?: string;
  isMostRecent: boolean;
  messageCompleted: boolean;
}) {
  const { type, content } = props.message;
  const responseFeedbackUrls =
    props.feedbackUrls?.[RESPONSE_FEEDBACK_KEY] ?? [];
  const sourceFeedbackUrls = props.feedbackUrls?.[SOURCE_CLICK_KEY] ?? [];
  const isUser = type === "human";
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [comment, setComment] = useState("");
  const upButtonRef = useRef(null);
  const downButtonRef = useRef(null);

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

  const sendUserFeedback = async (score: number) => {
    if (responseFeedbackUrls.length === 0) {
      return;
    }
    if (isLoading) {
      return;
    }
    setIsLoading(true);
    try {
      const feedbackResponses = [];
      for (const feedbackUrl of responseFeedbackUrls) {
        if (!hasAllowedURLPrefix(feedbackUrl)) {
          continue;
        }

        const data = await sendFeedback({
          feedbackUrl,
          score,
          feedbackId: feedback?.feedback_id,
          comment,
          isExplicit: true,
        });
        feedbackResponses.push(data);
      }
      if (feedbackResponses.every((response) => response.code === 200)) {
        setFeedback({ score, feedback_id: feedbackResponses[0].feedbackId });
        score == 1 ? animateButton("upButton") : animateButton("downButton");
        if (comment) {
          setComment("");
        }
      }
    } catch (e: any) {
      console.error("Error:", e);
      toast.error(e.message);
    }
    setIsLoading(false);
  };

  const sources = props.message.sources ?? [];
  const { filtered: filteredSources, indexMap: sourceIndexMap } =
    filterSources(sources);

  // Use an array of highlighted states as a state since React
  // complains when creating states in a loop
  const [highlighedSourceLinkStates, setHighlightedSourceLinkStates] = useState(
    filteredSources.map(() => false),
  );
  const answerElements =
    type === "ai"
      ? createAnswerElements(
          content,
          filteredSources,
          sourceIndexMap,
          highlighedSourceLinkStates,
          setHighlightedSourceLinkStates,
        )
      : [];

  const animateButton = (buttonId: string) => {
    let button: HTMLButtonElement | null;
    if (buttonId === "upButton") {
      button = upButtonRef.current;
    } else if (buttonId === "downButton") {
      button = downButtonRef.current;
    } else {
      return;
    }
    if (!button) return;
    let resolvedButton = button as HTMLButtonElement;
    resolvedButton.classList.add("animate-ping");
    setTimeout(() => {
      resolvedButton.classList.remove("animate-ping");
    }, 500);

    emojisplosion({
      emojiCount: 10,
      uniqueness: 1,
      position() {
        const offset = cumulativeOffset(button);

        return {
          x: offset.left + resolvedButton.clientWidth / 2,
          y: offset.top + resolvedButton.clientHeight / 2,
        };
      },
      emojis: buttonId === "upButton" ? ["👍"] : ["👎"],
    });
  };

  return (
    <VStack align="start" spacing={5} pb={5}>
      {!isUser && filteredSources.length > 0 && (
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
              <HStack spacing={"10px"} maxWidth={"100%"} overflow={"auto"}>
                {filteredSources.map((source, index) => (
                  <Box key={index} alignSelf={"stretch"} width={40}>
                    <SourceBubble
                      source={source}
                      highlighted={highlighedSourceLinkStates[index]}
                      onMouseEnter={() =>
                        setHighlightedSourceLinkStates(
                          filteredSources.map((_, i) => i === index),
                        )
                      }
                      onMouseLeave={() =>
                        setHighlightedSourceLinkStates(
                          filteredSources.map(() => false),
                        )
                      }
                      feedbackUrls={sourceFeedbackUrls}
                    />
                  </Box>
                ))}
              </HStack>
            </VStack>
          </Flex>

          <Heading size="lg" fontWeight="medium" color="blue.300">
            Answer
          </Heading>
        </>
      )}

      {isUser ? (
        <Heading size="lg" fontWeight="medium" color="white">
          {content}
        </Heading>
      ) : (
        <Box className="whitespace-pre-wrap" color="white">
          {answerElements}
        </Box>
      )}

      {props.message.type !== "human" &&
        props.isMostRecent &&
        props.messageCompleted && (
          <HStack spacing={2}>
            <Button
              ref={upButtonRef}
              size="sm"
              variant="outline"
              colorScheme={feedback === null ? "green" : "gray"}
              onClick={() => {
                if (feedback === null && responseFeedbackUrls) {
                  sendUserFeedback(1);
                  animateButton("upButton");
                } else {
                  toast.error("You have already provided your feedback.");
                }
              }}
            >
              👍
            </Button>
            <Button
              ref={downButtonRef}
              size="sm"
              variant="outline"
              colorScheme={feedback === null ? "red" : "gray"}
              onClick={() => {
                if (feedback === null && responseFeedbackUrls) {
                  sendUserFeedback(0);
                  animateButton("downButton");
                } else {
                  toast.error("You have already provided your feedback.");
                }
              }}
            >
              👎
            </Button>
          </HStack>
        )}

      {!isUser && <Divider mt={4} mb={4} />}
    </VStack>
  );
}
