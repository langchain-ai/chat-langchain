import { ExternalLink } from "lucide-react";
import { LangSmithSVG } from "./icons/langsmith";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";
import { useAssistantToolUI } from "@assistant-ui/react";
import { useCallback } from "react";

export const useLangSmithLinkToolUI = () =>
  useAssistantToolUI({
    toolName: "langsmith_tool_ui",
    render: useCallback((input) => {
      return (
        <TooltipIconButton
          tooltip="View run in LangSmith"
          variant="ghost"
          className="transition-colors w-4 h-3 ml-3 mt-2 mb-[-8px]"
          delayDuration={400}
          onClick={() => window.open(input.args.sharedRunURL, "_blank")}
        >
          <span className="flex flex-row items-center gap-1 w-11 h-7">
            <ExternalLink />
            <LangSmithSVG className="text-[#CA632B] hover:text-[#CA632B]/95" />
          </span>
        </TooltipIconButton>
      );
    }, []),
  });
