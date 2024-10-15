import { useThreadRuntime } from "@assistant-ui/react";
import { Card, CardTitle } from "./ui/card";

const suggestedQuestions = [
  "How do I use a RecursiveUrlLoader to load content from a page?",
  "How can I define the state schema for my LangGraph graph?",
  "How can I run a model locally on my laptop with Ollama?",
  "Explain RAG techniques and how LangGraph can implement them.",
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
    <div className="w-full flex flex-wrap gap-2">
      {suggestedQuestions.map((question, idx) => (
        <Card
          onClick={() => handleSend(question)}
          key={`suggested-question-${idx}`}
          className="w-[300px] h-[75px] bg-[#282828] border-gray-600 cursor-pointer transition-colors ease-in hover:bg-[#2b2b2b]"
        >
          <CardTitle className="mx-auto p-5 text-gray-200 font-normal">
            {question}
          </CardTitle>
        </Card>
      ))}
    </div>
  );
}
