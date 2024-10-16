import { useAssistantToolUI } from "@assistant-ui/react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { BookOpenText, Plus, SquareArrowOutUpRight } from "lucide-react";
import {
  TooltipProvider,
  TooltipTrigger,
  TooltipContent,
  Tooltip,
} from "./ui/tooltip";
import { Sheet, SheetContent, SheetTrigger } from "./ui/sheet";

type Document = {
  page_content: string;
  metadata: Record<string, any>;
};

const DocumentCard = ({ document }: { document: Document }) => {
  const description =
    document.metadata.description && document.metadata.description !== ""
      ? document.metadata.description
      : document.page_content.slice(0, 250);

  return (
    <Card className="md:w-[200px] sm:w-[200px] w-full h-[110px] bg-inherit border-gray-500 flex flex-col">
      <CardHeader className="flex-shrink-0 px-3 pt-2 pb-0">
        <CardTitle className="text-sm font-light text-gray-300 line-clamp-1 overflow-hidden p-[-24px]">
          {document.metadata.title}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col px-3 flex-grow justify-between">
        <p className="text-xs font-light text-gray-400 line-clamp-4 overflow-hidden">
          {description}
        </p>
      </CardContent>
    </Card>
  );
};

const DocumentCardTooltip = ({ document }: { document: Document }) => {
  const description =
    document.metadata.description && document.metadata.description !== ""
      ? document.metadata.description
      : document.page_content.slice(0, 250);

  return (
    <TooltipProvider>
      <Tooltip defaultOpen delayDuration={0}>
        <TooltipTrigger asChild>
          <DocumentCard document={document} />
        </TooltipTrigger>
        <TooltipContent className="flex flex-col max-w-[600px] whitespace-pre-wrap">
          <div className="flex flex-col gap-1">
            <p className="font-medium text-gray-300">
              {document.metadata.title}
            </p>
            <div className="flex flex-wrap justify-start">
              <a
                href={document.metadata.source}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-blue-400 hover:text-blue-300 transition-colors duration-200 break-all"
              >
                Source{" "}
                <SquareArrowOutUpRight className="ml-1 h-4 w-4 flex-shrink-0" />
              </a>
            </div>
            <p className="text-xs font-light text-gray-400">{description}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export const useSelectedDocumentsUI = () =>
  useAssistantToolUI({
    toolName: "selected_documents",
    render: (input) => {
      if (!input.args?.documents || input.args.documents.length === 0) {
        return null;
      }

      const documents = input.args.documents as Document[];
      const displayedDocuments = documents.slice(0, 3);
      const remainingDocuments = documents.slice(3);

      return (
        <div className="flex flex-col mb-4">
          <span className="flex flex-row gap-2 items-center justify-start pb-4 text-gray-300">
            <BookOpenText className="w-5 h-5" />
            <p className="text-xl">Selected Context</p>
          </span>
          <div className="flex flex-wrap items-center justify-start gap-2">
            {displayedDocuments.map((document, docIndex) => (
              <DocumentCardTooltip
                key={`final-document-${docIndex}`}
                document={document}
              />
            ))}
            {remainingDocuments.length > 0 && (
              <Sheet>
                <SheetTrigger>
                  <div className="flex items-center border-[1px] border-gray-500 justify-center w-[40px] h-[110px] bg-[#282828] hover:bg-[#2b2b2b] rounded-md cursor-pointer transition-colors duration-200">
                    <Plus className="w-6 h-6 text-gray-300" />
                  </div>
                </SheetTrigger>
                <SheetContent
                  side="right"
                  className="bg-[#282828] border-none overflow-y-auto flex-grow scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent md:min-w-[50vw] min-w-[70vw]"
                >
                  <div className="flex flex-col gap-4">
                    <h2 className="text-xl font-semibold text-gray-300">
                      All Selected Documents
                    </h2>
                    <div className="flex flex-wrap gap-2">
                      {documents.map((document, docIndex) => (
                        <DocumentCardTooltip
                          key={`all-documents-${docIndex}`}
                          document={document}
                        />
                      ))}
                    </div>
                  </div>
                </SheetContent>
              </Sheet>
            )}
          </div>
        </div>
      );
    },
  });
