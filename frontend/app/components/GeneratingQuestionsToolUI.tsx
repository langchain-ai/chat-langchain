import { useAssistantToolUI } from "@assistant-ui/react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { LoaderCircle, Globe, Plus } from "lucide-react";
import { DocumentDialog } from "./DocumentDialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { Sheet, SheetContent, SheetTrigger } from "./ui/sheet";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";
import { DocumentCard, Document } from "./DocumentCard";

type Question = {
  question: string;
  step: number;
  // Not rendered in the UI ATM.
  queries?: string[];
  documents?: Document[];
};

const QuestionCard = ({ question }: { question: Question }) => {
  const displayedDocuments = question.documents?.slice(0, 6) || [];
  const remainingDocuments = question.documents?.slice(6) || [];

  return (
    <Card className="md:w-[250px] sm:w-[250px] w-full md:max-w-full h-[140px] bg-inherit border-gray-500 flex flex-col gap-2">
      <CardHeader className="flex-shrink-0 px-3 pt-2 pb-0">
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
      <CardContent className="flex flex-col flex-grow px-3 pb-2 justify-between mt-auto">
        <div className="flex flex-col gap-1 mt-auto">
          <hr className="border-gray-400" />
          <div className="flex flex-wrap items-start justify-start gap-2 pt-1">
            {question.documents?.length ? (
              <>
                {displayedDocuments.map((doc: Document, idx: number) => (
                  <DocumentDialog
                    key={`document-${question.step}-${idx}`}
                    document={doc}
                  />
                ))}
                {remainingDocuments.length > 0 && (
                  <Sheet>
                    <SheetTrigger>
                      <TooltipIconButton
                        tooltip={`See ${remainingDocuments.length} more documents`}
                        variant="outline"
                        className="w-6 h-6 z-50 transition-colors ease-in-out bg-transparent hover:bg-gray-500 border-gray-400 text-gray-300"
                      >
                        <Plus />
                      </TooltipIconButton>
                    </SheetTrigger>
                    <SheetContent
                      side="right"
                      className="bg-[#282828] border-none overflow-y-auto flex-grow scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent md:min-w-[50vw] min-w-[70vw]"
                    >
                      <div className="flex flex-col gap-4">
                        <h2 className="text-xl font-semibold text-gray-300">
                          All Documents for Question
                        </h2>
                        <div className="flex flex-wrap gap-2">
                          {question.documents?.map(
                            (doc: Document, idx: number) => (
                              <DocumentDialog
                                key={`all-documents-${idx}`}
                                document={doc}
                                trigger={<DocumentCard document={doc} />}
                              />
                            ),
                          )}
                        </div>
                      </div>
                    </SheetContent>
                  </Sheet>
                )}
              </>
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
};

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
            <p className="text-xl">Research Plan & Sources</p>
          </span>
          <div className="mb-10">
            <div className="flex flex-wrap items-start justify-start gap-2">
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
