import { useAssistantToolUI } from "@assistant-ui/react";
import { BookOpenText, Plus } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "./ui/sheet";
import { DocumentDialog } from "./DocumentDialog";
import { DocumentCard, Document } from "./DocumentCard";
import { useCallback } from "react";

export const useSelectedDocumentsUI = () =>
  useAssistantToolUI({
    toolName: "selected_documents",
    // Wrap the component in a useCallback to keep the identity stable.
    // Allows the component to be interactable and not be re-rendered on every state change.
    render: useCallback((input) => {
      if (!input.args?.documents || input.args.documents.length === 0) {
        return null;
      }

      // Filter out duplicate documents
      const uniqueDocuments = (input.args.documents as Document[]).reduce(
        (acc, current) => {
          const x = acc.find(
            (item) => item.page_content === current.page_content,
          );
          if (!x) {
            return acc.concat([current]);
          } else {
            return acc;
          }
        },
        [] as Document[],
      );

      const displayedDocuments = uniqueDocuments.slice(0, 3);
      const remainingDocuments = uniqueDocuments.slice(3);

      return (
        <div className="flex flex-col mb-4">
          <span className="flex flex-row gap-2 items-center justify-start pb-4 text-gray-300">
            <BookOpenText className="w-5 h-5" />
            <p className="text-xl">Selected Context</p>
          </span>
          <div className="flex flex-wrap items-center justify-start gap-2">
            {displayedDocuments.map((document, docIndex) => (
              <DocumentDialog
                key={`all-documents-${docIndex}`}
                document={document}
                trigger={<DocumentCard document={document} />}
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
                      {uniqueDocuments.map((document, docIndex) => (
                        <DocumentDialog
                          key={`all-documents-${docIndex}`}
                          document={document}
                          trigger={<DocumentCard document={document} />}
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
    }, []),
  });
