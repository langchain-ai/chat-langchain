import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { emojisplosion } from "emojisplosion";
import { useState } from "react";
import { SourceBubble, Source } from './SourceBubble';
import { Flex, Spacer, Box, Heading, HStack, VStack, Divider, Text} from '@chakra-ui/react'

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
  const colorClassName =
    props.message.role === "user" ? "bg-sky-600" : "bg-slate-50 text-black";
  const alignmentClassName =
    props.message.role === "user" ? "ml-auto" : "mr-auto";
  const urlDelimiter = "SOURCES:----------------------------";

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

  function parseUrls(text:string) {
    if (!text.includes('SOURCES:----------------------------')) {
      return [];
    }
    const parts = text.split('SOURCES:----------------------------');
    
    if (parts.length <1) {
      return [];
    }
  
    let urls = parts[0].trim().split('\n');

    let sources = urls.map((url) => {
      let urlParts = url.split('"');
      let titleParts = url.split(':');
      return {url: urlParts[1], title: titleParts[0]};
    });
    
    return sources;
  }

  const sources = parseUrls(props.message.content);
  const messageParts = props.message.content.split(urlDelimiter);
  const aiResponse = messageParts.length > 1 ? messageParts[1] : props.message.content.includes(urlDelimiter) ? "" : props.message.content;

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
              <Box key={index}><SourceBubble source={source}/></Box>
            ))
            }
          </HStack> 
        </VStack>
      </Flex>

      <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"blue.300"} paddingTop={"20px"}>Answer</Heading></>)
}
      {isUser ? <Heading size={"lg"} fontWeight={"medium"} color={"white"}>{aiResponse}</Heading> : <div
          className="whitespace-pre-wrap"
          style={{"color": "white"}}
          dangerouslySetInnerHTML={{ __html: aiResponse }}
        ></div>}
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
