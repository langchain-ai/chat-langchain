import { useAssistantToolUI } from "@assistant-ui/react";
import NextImage from "next/image";

export const useAnswerHeaderToolUI = () =>
  useAssistantToolUI({
    toolName: "answer_header",
    render: (_) => {
      return (
        <div className="flex flex-row gap-2 items-center justify-start pb-4 text-black-300">
          <NextImage
              src="public/images/answer_logo.svg"
              className=""
              alt="Answer Logo"
              width={10}
              height={10}
          />
          <p className="text-2xl font-bold">Answer</p>
        </div>
      );
    },
  });
