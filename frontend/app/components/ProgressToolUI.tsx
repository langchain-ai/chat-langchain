import { useAssistantToolUI } from "@assistant-ui/react";
import { Progress } from "./ui/progress";
import { cn } from "../utils/cn";

export const stepToProgressFields = (step: number) => {
  switch (step) {
    case 0:
      return {
        text: "Routing Query",
        progress: 20,
      };
    case 1:
      return {
        text: "Generating questions",
        progress: 40,
      };
    case 2:
      return {
        text: "Doing research",
        progress: 65,
      };
    // case 2.25:
    //   return {
    //     text: "Doing research (2/3)",
    //     progress: 65,
    //   };
    // case 2.5:
    //   return {
    //     text: "Doing research (3/3)",
    //     progress: 75,
    //   };
    case 3:
      return {
        text: "Generating answer",
        progress: 85,
      };
    case 4:
      return {
        text: "Done",
        progress: 100,
      };
    default:
      return {
        text: "Working on it",
        progress: 0,
      };
  }
};

export const useProgressToolUI = () =>
  useAssistantToolUI({
    toolName: "progress",
    render: (input) => {
      const { text, progress } = stepToProgressFields(input.args.step);

      return (
        <div className="flex flex-row w-full min-w-[600px] items-center justify-start gap-3">
          <Progress
            value={progress}
            indicatorClassName="bg-gray-700"
            className="w-[60%]"
          />
          <p
            className={cn(
              "text-gray-500 mt-2 text-sm font-light",
              progress !== 100 ? "animate-pulse" : "",
            )}
          >
            {text}
          </p>
        </div>
      );
    },
  });
