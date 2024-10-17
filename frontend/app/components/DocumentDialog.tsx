import { SquareArrowOutUpRight, File } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../components/ui/dialog";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";

interface DocumentDialogProps {
  document: Record<string, any>;
  trigger?: React.ReactNode;
}

export function DocumentDialog(props: DocumentDialogProps) {
  const trigger = props.trigger || (
    <TooltipIconButton
      tooltip={props.document.metadata.title}
      variant="outline"
      className="w-6 h-6 z-50 transition-colors ease-in-out bg-transparent hover:bg-gray-500 border-gray-400 text-gray-300"
    >
      <File />
    </TooltipIconButton>
  );

  return (
    <Dialog>
      <DialogTrigger asChild={!props.trigger}>{trigger}</DialogTrigger>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto bg-gray-700 text-gray-200">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-start gap-4">
            <p className="text-gray-100 break-words">
              {props.document.metadata.title}
            </p>
            <div className="flex flex-wrap justify-start">
              <a
                href={props.document.metadata.source}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-blue-400 hover:text-blue-300 transition-colors duration-200 break-all"
              >
                Source{" "}
                <SquareArrowOutUpRight className="ml-1 h-4 w-4 flex-shrink-0" />
              </a>
            </div>
          </DialogTitle>
        </DialogHeader>
        <DialogDescription className="text-gray-300 break-words whitespace-normal">
          {props.document.metadata.description}
        </DialogDescription>

        <hr />
        <div className="mt-2 overflow-hidden">
          <p className="whitespace-pre-wrap text-gray-200 break-words overflow-wrap-anywhere">
            {props.document.page_content}
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
