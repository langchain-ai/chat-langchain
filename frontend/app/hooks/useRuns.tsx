import { useCallback } from "react";
import { Feedback } from "langsmith";

export interface FeedbackResponse {
  success: boolean;
  feedback: Feedback;
}

export function useRuns() {
  /**
   * Generates a public shared run ID for the given run ID.
   */
  const shareRun = async (runId: string): Promise<string | undefined> => {
    const res = await fetch("/api/runs/share", {
      method: "POST",
      body: JSON.stringify({ runId }),
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!res.ok) {
      return;
    }

    const { sharedRunURL } = await res.json();
    return sharedRunURL;
  };

  const sendFeedback = useCallback(
    async (
      runId: string,
      feedbackKey: string,
      score: number,
      comment?: string,
    ): Promise<FeedbackResponse | undefined> => {
      try {
        const res = await fetch("/api/runs/feedback", {
          method: "POST",
          body: JSON.stringify({ runId, feedbackKey, score, comment }),
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!res.ok) {
          return;
        }

        return (await res.json()) as FeedbackResponse;
      } catch (error) {
        console.error("Error sending feedback:", error);
        return;
      }
    },
    [],
  );

  return {
    shareRun,
    sendFeedback,
  };
}
