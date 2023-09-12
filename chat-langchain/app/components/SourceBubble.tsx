import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { emojisplosion } from "emojisplosion";
import { useState } from "react";
import { ChakraProvider, Card, CardHeader, CardBody, CardFooter, Text, Heading, Image, Box} from '@chakra-ui/react'

export type Source = {
  url: string;
  title: string;
};

export function SourceBubble(props: {
  source: Source;
}) {
  const [feedbackColor, setFeedbackColor] = useState("");
  const [isMouseOver, setIsMouseOver] = useState(false);

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

  console.log(props.source.url)

  return (
        <Card onClick={() => {
            window.open(props.source.url, "_blank");
        }}
        backgroundColor={isMouseOver ? "rgb(78,78,81)" : "rgb(58, 58, 61)"}
        onMouseOver={() => {setIsMouseOver(true)}}
        onMouseLeave={() => {setIsMouseOver(false)}}
        cursor={"pointer"}
        alignSelf={"stretch"}
        height="100%"
        overflow={"hidden"}>
            <CardBody><Heading size={"sm"} fontWeight={"normal"} color={"white"}>{props.source.title}</Heading></CardBody>
        </Card>
  );
}
