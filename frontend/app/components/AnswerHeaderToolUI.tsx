import { useAssistantToolUI } from "@assistant-ui/react";
import { BrainCog } from "lucide-react";

export const useAnswerHeaderToolUI = () =>
  useAssistantToolUI({
    toolName: "answer_header",
    render: (_) => {
      return (
        <div className="flex flex-row gap-2 items-center justify-start pb-4 text-gray-300">
          <BrainCog className="w-5 h-5" />
          <p className="text-xl">Answer</p>
        </div>
      );
    },
  });
