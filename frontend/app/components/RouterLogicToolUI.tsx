import { useAssistantToolUI } from "@assistant-ui/react";
import { Route } from "lucide-react";

export const useRouterLogicUI = () =>
  useAssistantToolUI({
    toolName: "router_logic",
    render: (input) => {
      if (!input.args?.logic || input.args.logic === 0) {
        return null;
      }

      return (
        <div className="flex flex-col mb-4">
          <span className="flex flex-row gap-2 items-center justify-start pb-0 text-gray-300">
            <Route className="w-5 h-5" />
            <p className="text-xl">Router Logic</p>
          </span>
          <p className="text-sm text-gray-400">{input.args.logic}</p>
        </div>
      );
    },
  });
