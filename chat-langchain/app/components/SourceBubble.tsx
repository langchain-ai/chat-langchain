import 'react-toastify/dist/ReactToastify.css';
import { emojisplosion } from "emojisplosion";
import { Card, CardBody, Heading } from '@chakra-ui/react'

export type Source = {
  url: string;
  title: string;
};

export function SourceBubble(props: {
  source: Source;
  highlighted: boolean;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
}) {
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
        <Card onClick={() => {
            window.open(props.source.url, "_blank");
        }}
        backgroundColor={props.highlighted ? "rgb(58, 58, 61)" : "rgb(78,78,81)"}
        onMouseEnter={props.onMouseEnter}
        onMouseLeave={props.onMouseLeave}
        cursor={"pointer"}
        alignSelf={"stretch"}
        height="100%"
        overflow={"hidden"}>
            <CardBody><Heading size={"sm"} fontWeight={"normal"} color={"white"}>{props.source.title}</Heading></CardBody>
        </Card>
  );
}