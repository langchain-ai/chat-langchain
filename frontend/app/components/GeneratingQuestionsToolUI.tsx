import { useAssistantToolUI } from "@assistant-ui/react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LoaderCircle } from "lucide-react";
import { Document } from "@langchain/core/documents";
import { DocumentDialog } from "./DocumentDialog";

export const useGeneratingQuestionsUI = () =>
  useAssistantToolUI({
    toolName: "generating_questions",
    render: (input) => {
      if (!input.args?.question) {
        return null;
      }

      if (input.args.documents) {
        console.log("Gots docs!", input.args.documents.length);
      }

      return (
        <Card className="w-[500px] bg-black text-white">
          <CardHeader>
            <CardTitle>{input.args.question}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex flex-col items-start justify-start w-full">
              <p className="text-sm font-semibold">Sub-queries:</p>
              {input.args.queries?.length ? (
                <ul className="w-full space-y-2">
                  {input.args.queries.map((query: string, idx: number) => (
                    <li
                      key={`query-${input.args.step}-${idx}`}
                      className="bg-gray-800 rounded-md p-2 text-sm"
                    >
                      {query}
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="flex items-center justify-start gap-2">
                  <p>Generating queries</p>
                  <LoaderCircle className="animate-spin" />
                </span>
              )}
            </div>
            <div className="flex flex-col gap-1">
              <p className="text-sm text-left font-semibold">Documents:</p>
              <div className="flex items-center justify-start gap-5">
                {input.args?.documents?.length ? (
                  input.args.documents?.map((doc: Document, idx: number) => (
                    <DocumentDialog
                      key={`document-${input.args.step}-${idx}`}
                      document={doc}
                    />
                  ))
                ) : (
                  <span className="flex items-center justify-start gap-2">
                    <p>Finding documents</p>
                    <LoaderCircle className="animate-spin" />
                  </span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      );
    },
  });
