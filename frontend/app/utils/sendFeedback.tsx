import { v4 as uuidv4 } from "uuid";

type SendFeedbackProps = {
  // NOTE: feedback URL is a signed langsmith URL that already includes the run information & key for the feedback
  feedbackUrl: string;
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
  feedbackUrl,
  score,
  value,
  comment,
  feedbackId,
  isExplicit = true,
}: SendFeedbackProps) => {
  const feedback_id = feedbackId ?? uuidv4();
  const response = await fetch(feedbackUrl, {
    method: feedbackId ? "PATCH" : "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      score,
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
