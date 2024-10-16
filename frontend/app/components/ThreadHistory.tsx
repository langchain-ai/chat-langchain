import { isToday, isYesterday, isWithinInterval, subDays } from "date-fns";
import { dummyThreads } from "../utils/dummy";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";
import { Button } from "./ui/button";
import { SquarePen, History } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "./ui/sheet";

interface ThreadHistoryProps {
  assistantId: string | undefined;
}

interface ThreadProps {
  id: string;
  onClick: (id: string) => void;
  label: string;
  createdAt: Date;
}

const Thread = (props: ThreadProps) => (
  <Button
    className="px-2 hover:bg-[#393939] hover:text-white justify-start"
    size="sm"
    variant="ghost"
    onClick={() => props.onClick(props.id)}
  >
    <p className="truncate ... text-sm font-light">{props.label}</p>
  </Button>
);

const groupThreads = (threads: ThreadProps[]) => {
  const today = new Date();
  const yesterday = subDays(today, 1);
  const sevenDaysAgo = subDays(today, 7);

  return {
    today: threads.filter((thread) => isToday(thread.createdAt)),
    yesterday: threads.filter((thread) => isYesterday(thread.createdAt)),
    lastSevenDays: threads.filter((thread) =>
      isWithinInterval(thread.createdAt, {
        start: sevenDaysAgo,
        end: yesterday,
      }),
    ),
    older: threads.filter((thread) => thread.createdAt < sevenDaysAgo),
  };
};

const prettifyDateLabel = (group: string): string => {
  switch (group) {
    case "today":
      return "Today";
    case "yesterday":
      return "Yesterday";
    case "lastSevenDays":
      return "Last 7 days";
    case "older":
      return "Older";
    default:
      return group;
  }
};

interface ThreadsListProps {
  groupedThreads: {
    today: ThreadProps[];
    yesterday: ThreadProps[];
    lastSevenDays: ThreadProps[];
    older: ThreadProps[];
  };
}

function ThreadsList(props: ThreadsListProps) {
  return (
    <div className="flex flex-col px-3 pt-3 gap-4">
      {Object.entries(props.groupedThreads).map(([group, threads]) =>
        threads.length > 0 ? (
          <div key={group}>
            <h3 className="text-sm font-medium text-gray-400 mb-1 pl-2">
              {prettifyDateLabel(group)}
            </h3>
            <div className="flex flex-col gap-1">
              {threads.map((thread) => (
                <Thread key={thread.id} {...thread} />
              ))}
            </div>
          </div>
        ) : null,
      )}
    </div>
  );
}

export function ThreadHistory(props: ThreadHistoryProps) {
  const groupedThreads = groupThreads(dummyThreads);

  return (
    <span>
      {/* Tablet & up */}
      <div className="hidden md:flex flex-col w-[260px] h-full">
        <div className="flex-grow border-l-[1px] border-[#393939] my-6">
          <div className="flex flex-row items-center justify-between border-b-[1px] border-[#393939] pt-3 px-2 mx-4 -mt-4 text-gray-200">
            <p className="text-lg font-medium">Chat History</p>
            <TooltipIconButton
              tooltip="New chat"
              variant="ghost"
              className="w-fit p-2"
            >
              <SquarePen className="w-5 h-5" />
            </TooltipIconButton>
          </div>
          <ThreadsList groupedThreads={groupedThreads} />
        </div>
      </div>
      {/* Mobile */}
      <span className="md:hidden flex flex-row gap-2 mt-2 mr-2">
        <Sheet>
          <SheetTrigger>
            <TooltipIconButton
              tooltip="New chat"
              variant="ghost"
              className="w-fit p-2"
            >
              <History className="w-6 h-6" />
            </TooltipIconButton>
          </SheetTrigger>
          <SheetContent className="bg-[#282828] border-none">
            <ThreadsList groupedThreads={groupedThreads} />
          </SheetContent>
        </Sheet>
        <TooltipIconButton
          tooltip="New chat"
          variant="ghost"
          className="w-fit p-2"
        >
          <SquarePen className="w-6 h-6" />
        </TooltipIconButton>
      </span>
    </span>
  );
}
