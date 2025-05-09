import { useAssistantToolUI } from "@assistant-ui/react";
import { BrainCog } from "lucide-react";

export const useAnswerHeaderToolUI = () =>
  useAssistantToolUI({
    toolName: "answer_header",
    render: (_) => {
      return (
        <div className="flex flex-row gap-2 items-center justify-start pb-4 text-black-300">
          <BrainCog className="w-6 h-6" />
          <p className="text-2xl font-bold">Answer</p>
        </div>
      );
    },
  });
