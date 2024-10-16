import { useAssistantToolUI } from "@assistant-ui/react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { BookOpenText, SquareArrowOutUpRight } from "lucide-react";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "./ui/carousel";
import {
  TooltipProvider,
  TooltipTrigger,
  TooltipContent,
  Tooltip,
} from "./ui/tooltip";
import { useState } from "react";

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
    <Card className="w-[200px] h-[110px] bg-inherit border-gray-500 flex flex-col">
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
  const [isOpen, setIsOpen] = useState(true);

  const description =
    document.metadata.description && document.metadata.description !== ""
      ? document.metadata.description
      : document.page_content.slice(0, 250);

  return (
    <TooltipProvider>
      <Tooltip
        defaultOpen
        delayDuration={0}
        open={isOpen}
        onOpenChange={setIsOpen}
      >
        <TooltipTrigger asChild>
          <div
            onMouseEnter={() => {
              console.log("Mouse enter");
              setIsOpen(true);
            }}
            onMouseLeave={() => {
              console.log("Mouse leave");
              setIsOpen(false);
            }}
          >
            <DocumentCard document={document} />
          </div>
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

      return (
        <div className="flex flex-col mb-4">
          <span className="flex flex-row gap-2 items-center justify-start pb-4 text-gray-300">
            <BookOpenText className="w-5 h-5" />
            <p className="text-xl">Selected Context</p>
          </span>
          <Carousel
            opts={{
              align: "start",
            }}
            className="mb-10 w-fit max-w-3xl"
          >
            <CarouselContent className="-ml-[0px]">
              {(input.args.documents as Document[]).map(
                (document, docIndex) => (
                  <CarouselItem
                    key={`final-document-${docIndex}`}
                    className="pl-[0px] md:basis-[30%] lg:basis-[28%]"
                  >
                    <DocumentCardTooltip document={document} />
                  </CarouselItem>
                ),
              )}
            </CarouselContent>
            <CarouselPrevious />
            <CarouselNext />
          </Carousel>
        </div>
      );
    },
  });
