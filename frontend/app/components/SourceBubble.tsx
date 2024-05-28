import "react-toastify/dist/ReactToastify.css";
import { Card, CardBody, Heading } from "@chakra-ui/react";
import { sendFeedback } from "../utils/sendFeedback";
import { Source } from "../types";

export function SourceBubble({
  source,
  highlighted,
  onMouseEnter,
  onMouseLeave,
  feedbackUrl,
}: {
  source: Source;
  highlighted: boolean;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
  feedbackUrl?: string;
}) {
  return (
    <Card
      onClick={async () => {
        window.open(source.url, "_blank");
        if (feedbackUrl) {
          await sendFeedback({
            feedbackUrl,
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
