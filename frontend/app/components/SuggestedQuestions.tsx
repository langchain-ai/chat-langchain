import { useThreadRuntime } from "@assistant-ui/react";
import NextImage from "next/image";

const suggestedQuestions = [
  "What does verafiles do?",
  "How does verafiles conduct their fact checks?",
  "Is fact checking hard?",
];

export function SuggestedQuestions() {
  const threadRuntime = useThreadRuntime();

  const handleSend = (text: string) => {
    threadRuntime.append({
      role: "user",
      content: [{ type: "text", text }],
    });
  };

  return (
    <div className="w-full grid grid-cols-1 border border-[#A5A5A5] rounded-2xl mt-4 mb-16 p-4">
      {suggestedQuestions.map((question, idx) => (
        <div
          onClick={() => handleSend(question)}
          key={`suggested-question-${idx}`}
          className="w-full border-gray-600 cursor-pointer"
        >
          <p className="flex flex-row gap-2 px-4 py-1.5 text-[#0F5579] text-lg font-bold transition-colors ease-in hover:text-[#2891E0]">
            {question}
            <NextImage
                src="/images/tabler_search.svg"
                className="rounded-3xl"
                alt="LangChain Logo"
                width={32}
                height={32}
                style={{ width: "auto", height: "auto" }}
              />
          </p>
        </div>
      ))}
    </div>
  );
}
