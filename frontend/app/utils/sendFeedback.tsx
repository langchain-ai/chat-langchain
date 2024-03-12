import { v4 as uuidv4 } from "uuid";
import { apiBaseUrl } from "./constants";

type SendFeedbackProps = {
  key: string;
  runId: string;
  score?: number;
  value?: string;
  comment?: string;
  feedbackId?: string;
  isExplicit: boolean;
};

type FeedbackResponse = {
  feedbackId: string;
  code: number;
  result: string;
};
export const sendFeedback = async ({
  score,
  key,
  runId,
  value,
  comment,
  feedbackId,
  isExplicit = true,
}: SendFeedbackProps) => {
  const feedback_id = feedbackId ?? uuidv4();
  const response = await fetch(apiBaseUrl + "/feedback", {
    method: feedbackId ? "PATCH" : "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      score,
      run_id: runId,
      key,
      value,
      feedback_id,
      comment,
      source_info: {
        is_explicit: isExplicit,
      },
    }),
  });
  const data = await response.json();
  return {
    ...data,
    feedbackId: feedback_id,
  } as FeedbackResponse;
};
