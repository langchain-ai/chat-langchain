import { useAssistantToolUI } from "@assistant-ui/react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { LoaderCircle, Globe } from "lucide-react";
import { DocumentDialog } from "./DocumentDialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

type Question = {
  question: string;
  step: number;
  // Not rendered in the UI ATM.
  queries?: string[];
  documents?: Record<string, any>[];
};

const QuestionCard = ({ question }: { question: Question }) => (
  <Card className="w-[250px] h-[180px] bg-inherit border-gray-500 flex flex-col">
    <CardHeader className="flex-shrink-0">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <CardTitle className="text-sm font-light text-gray-300 line-clamp-4 overflow-hidden">
              {question.question}
            </CardTitle>
          </TooltipTrigger>
          <TooltipContent className="max-w-[600px] whitespace-pre-wrap">
            <p>{question.question}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </CardHeader>
    <CardContent className="flex flex-col flex-grow justify-between">
      <div className="flex flex-col gap-1 mt-auto">
        <hr className="mb-2" />
        <div className="flex items-center justify-start gap-1">
          {question.documents?.length ? (
            question.documents?.map((doc: Record<string, any>, idx: number) => (
              <DocumentDialog
                key={`document-${question.step}-${idx}`}
                document={doc}
              />
            ))
          ) : (
            <span className="flex items-center justify-start gap-2 text-gray-400">
              <p className="text-sm">Finding documents</p>
              <LoaderCircle className="animate-spin w-4 h-4" />
            </span>
          )}
        </div>
      </div>
    </CardContent>
  </Card>
);

export const useGeneratingQuestionsUI = () =>
  useAssistantToolUI({
    toolName: "generating_questions",
    render: (input) => {
      if (!input.args?.questions || input.args.questions.length === 0) {
        return null;
      }

      return (
        <div className="flex flex-col mb-4">
          <span className="flex flex-row gap-2 items-center justify-start pb-4 text-gray-300">
            <Globe className="w-5 h-5" />
            <p className="text-xl">Generated Questions & Sources</p>
          </span>
          <div className="relative left-1/2 -translate-x-1/2 w-screen max-w-[70vw] mb-10">
            <div className="flex items-center justify-center gap-2">
              <div className="flex flex-row gap-3 items-center justify-center"></div>
              {(input.args.questions as Question[]).map(
                (question, questionIndex) => (
                  <QuestionCard
                    key={`question-${questionIndex}`}
                    question={question}
                  />
                ),
              )}
            </div>
          </div>
        </div>
      );
    },
  });
