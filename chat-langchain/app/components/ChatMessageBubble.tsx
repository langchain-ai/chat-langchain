import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { emojisplosion } from "emojisplosion";
import { useState } from "react";
import { SourceBubble, Source } from './SourceBubble';
import { Flex, Box, Heading, HStack, VStack, Divider} from '@chakra-ui/react'
import { InlineCitation } from './InlineCitation';

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
  const isUser = props.message.role === "user";
  const urlDelimiter = "SOURCES:----------------------------";

  const [feedbackColor, setFeedbackColor] = useState("");

  const cumulativeOffset = function(element: HTMLElement | null) {
      var top = 0, left = 0;
      do {
          top += element?.offsetTop  || 0;
          left += element?.offsetLeft || 0;
          element = (element?.offsetParent as HTMLElement) || null;
      } while (element);

      return {
          top: top,
          left: left
      };
  };

  function parseUrls(text: string) {
    if (!text.includes(urlDelimiter)) {
      return [];
    }
    const parts = text.split(urlDelimiter);

    if (parts.length <1) {
      return [];
    }

    let urls = parts[0].trim().split('\n');

    let sources: Source[] = urls.map((url) => {
      let urlParts = url.split('"');
      let titleParts = url.split(':');
      let title = titleParts[0].split(" |")[0];
      return {url: urlParts[1], title: title};
    });

    return sources;
  }

  const sources = parseUrls(props.message.content);

  // Use an array of highlighted states as a state since React
  // complains when creating states in a loop
  const [highlighedSourceLinkStates, setHighlightedSourceLinkStates] = useState(sources.map(() => false));
  const messageParts = props.message.content.split(urlDelimiter);
  let standaloneMessage = messageParts.length > 1
    ? messageParts.slice(-1)[0]
    : (props.message.content.includes(urlDelimiter) ? "" : props.message.content);

  const matches = Array.from(standaloneMessage.matchAll(/\[(\d+)\]/g));
  const answerElements = [];

  let previousSliceIndex = 0;
  for (const match of matches) {
    const sourceNumber = parseInt(match[1], 10);
    if (match.index && sources[sourceNumber]) {
      answerElements.push(<span
        key={`content:${previousSliceIndex}`}
        dangerouslySetInnerHTML={{__html: standaloneMessage.slice(previousSliceIndex, match.index)}}
      ></span>);
      answerElements.push(<InlineCitation
        key={`citation:${previousSliceIndex}`}
        source={sources[sourceNumber]}
        sourceNumber={sourceNumber}
        highlighted={highlighedSourceLinkStates[sourceNumber]}
        onMouseEnter={() => setHighlightedSourceLinkStates(sources.map((_, i) => i === sourceNumber))}
        onMouseLeave={() => setHighlightedSourceLinkStates(sources.map(() => false))}
      ></InlineCitation>);
      previousSliceIndex = match.index + match[0].length;
    }
  }
  answerElements.push(<span
    key={`content:${previousSliceIndex}`}
    dangerouslySetInnerHTML={{__html: standaloneMessage.slice(previousSliceIndex)}}
  ></span>);

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
            <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"blue.300"} paddingBottom={"10px"}>Sources</Heading>
            <HStack spacing={'10px'}>
              {
              sources.map((source, index) => (
                <Box key={index} alignSelf={"stretch"} width={40}>
                  <SourceBubble source={source}
                    highlighted={highlighedSourceLinkStates[index]}
                    onMouseEnter={() => setHighlightedSourceLinkStates(sources.map((_, i) => i === index))}
                    onMouseLeave={() => setHighlightedSourceLinkStates(sources.map(() => false))}
                  />
                </Box>
              ))
              }
            </HStack>
          </VStack>
        </Flex>

        <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"blue.300"} paddingTop={"20px"}>Answer</Heading>
        </>
      )}
      {isUser ? <Heading size={"lg"} fontWeight={"medium"} color={"white"}>{standaloneMessage}</Heading> : <div
          className="whitespace-pre-wrap"
          style={{"color": "white"}}
        >{answerElements}</div>}
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

      {!isUser && <Divider marginTop={"20px"} marginBottom={"20px"}/>}
    </VStack>
  );
}
