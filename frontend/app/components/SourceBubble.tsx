import "react-toastify/dist/ReactToastify.css";
import { Card, CardBody, Heading } from "@chakra-ui/react";
import { sendFeedback } from "../utils/sendFeedback";

export type Source = {
  url: string;
  title: string;
};

export function SourceBubble({
  source,
  highlighted,
  onMouseEnter,
  onMouseLeave,
  runId,
}: {
  source: Source;
  highlighted: boolean;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
  runId?: string;
}) {
  return (
    <Card
      onClick={async () => {
        window.open(source.url, "_blank");
        if (runId) {
          await sendFeedback({
            key: "user_click",
            runId,
            value: source.url,
            isExplicit: false,
          });
        }
      }}
      backgroundColor={highlighted ? "rgb(58, 58, 61)" : "rgb(78,78,81)"}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      cursor={"pointer"}
      alignSelf={"stretch"}
      height="100%"
      overflow={"hidden"}
    >
      <CardBody>
        <Heading size={"sm"} fontWeight={"normal"} color={"white"}>
          {source.title}
        </Heading>
      </CardBody>
    </Card>
  );
}
