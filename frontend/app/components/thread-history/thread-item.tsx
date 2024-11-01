import { useState } from "react";
import { Button } from "../ui/button";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { Trash2 } from "lucide-react";

export interface ThreadProps {
  id: string;
  onClick: () => void;
  onDelete: () => void;
  label: string;
  createdAt: Date;
}

export function Thread(props: ThreadProps) {
  const [isHovering, setIsHovering] = useState(false);

  return (
    <div
      className="flex flex-row gap-0 items-center justify-start w-full"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <Button
        className="px-2 hover:bg-[#393939] hover:text-white justify-start items-center flex-grow min-w-[191px] pr-0"
        size="sm"
        variant="ghost"
        onClick={props.onClick}
      >
        <p className="truncate text-sm font-light w-full text-left">
          {props.label}
        </p>
      </Button>
      {isHovering && (
        <TooltipIconButton
          tooltip="Delete thread"
          variant="ghost"
          className="hover:bg-[#373737] flex-shrink-0 p-2"
          onClick={props.onDelete}
        >
          <Trash2 className="w-4 h-4 text-[#575757] hover:text-red-500 transition-colors ease-in" />
        </TooltipIconButton>
      )}
    </div>
  );
}
