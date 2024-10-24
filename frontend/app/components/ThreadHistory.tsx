import { isToday, isYesterday, isWithinInterval, subDays } from "date-fns";
import { TooltipIconButton } from "./ui/assistant-ui/tooltip-icon-button";
import { Button } from "./ui/button";
import { SquarePen, History, Trash2 } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "./ui/sheet";
import { ThreadActual } from "../hooks/useThreads";
import { useToast } from "../hooks/use-toast";
import { Skeleton } from "./ui/skeleton";
import { useEffect, useState } from "react";

interface ThreadHistoryProps {
  isUserThreadsLoading: boolean;
  isEmpty: boolean;
  currentThread: string | undefined;
  userThreads: ThreadActual[];
  userId: string | undefined;
  createThread: (id: string) => Promise<any>;
  clearMessages: () => void;
  switchSelectedThread: (thread: ThreadActual) => void;
  getUserThreads: (id: string) => Promise<void>;
  deleteThread: (id: string) => Promise<void>;
}

interface ThreadProps {
  id: string;
  onClick: () => void;
  onDelete: () => void;
  label: string;
  createdAt: Date;
}

const Thread = (props: ThreadProps) => {
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
};

const LoadingThread = () => <Skeleton className="w-full h-8 bg-[#373737]" />;

const convertThreadActualToThreadProps = (
  thread: ThreadActual,
  switchSelectedThread: (thread: ThreadActual) => void,
  deleteThread: (id: string) => void,
): ThreadProps => ({
  id: thread.thread_id,
  label: thread.values?.messages?.[0].content || "Untitled",
  createdAt: new Date(thread.created_at),
  onClick: () => {
    return switchSelectedThread(thread);
  },
  onDelete: () => {
    return deleteThread(thread.thread_id);
  },
});

const groupThreads = (
  threads: ThreadActual[],
  switchSelectedThread: (thread: ThreadActual) => void,
  deleteThread: (id: string) => void,
) => {
  const today = new Date();
  const yesterday = subDays(today, 1);
  const sevenDaysAgo = subDays(today, 7);

  return {
    today: threads
      .filter((thread) => isToday(new Date(thread.created_at)))
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
      .map((t) =>
        convertThreadActualToThreadProps(t, switchSelectedThread, deleteThread),
      ),
    yesterday: threads
      .filter((thread) => isYesterday(new Date(thread.created_at)))
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
      .map((t) =>
        convertThreadActualToThreadProps(t, switchSelectedThread, deleteThread),
      ),
    lastSevenDays: threads
      .filter((thread) =>
        isWithinInterval(new Date(thread.created_at), {
          start: sevenDaysAgo,
          end: yesterday,
        }),
      )
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
      .map((t) =>
        convertThreadActualToThreadProps(t, switchSelectedThread, deleteThread),
      ),
    older: threads
      .filter((thread) => new Date(thread.created_at) < sevenDaysAgo)
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
      .map((t) =>
        convertThreadActualToThreadProps(t, switchSelectedThread, deleteThread),
      ),
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
  const { toast } = useToast();
  const groupedThreads = groupThreads(
    props.userThreads,
    props.switchSelectedThread,
    props.deleteThread,
  );

  const createThread = async () => {
    if (!props.userId) {
      toast({
        title: "Error creating thread",
        description: "Your user ID was not found. Please try again later.",
      });
      return;
    }
    const currentThread = props.userThreads.find(
      (thread) => thread.thread_id === props.currentThread,
    );
    if (currentThread && !currentThread.values && props.isEmpty) {
      return;
    }

    props.clearMessages();
    await props.createThread(props.userId);
    // Re-fetch threads so that the new thread shows up.
    await props.getUserThreads(props.userId);
  };

  return (
    <span>
      {/* Tablet & up */}
      <div className="hidden md:flex flex-col w-[260px] h-full">
        <div className="flex-grow border-r-[1px] border-[#393939] my-6 flex flex-col overflow-hidden">
          <div className="flex flex-row items-center justify-between border-b-[1px] border-[#393939] pt-3 px-2 mx-4 -mt-4 text-gray-200">
            <p className="text-lg font-medium">Chat History</p>
            {props.userId ? (
              <TooltipIconButton
                tooltip="New chat"
                variant="ghost"
                className="w-fit p-2"
                onClick={createThread}
              >
                <SquarePen className="w-5 h-5" />
              </TooltipIconButton>
            ) : null}
          </div>
          <div className="overflow-y-auto flex-grow scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
            {props.isUserThreadsLoading && !props.userThreads.length ? (
              <div className="flex flex-col gap-1 px-3 pt-3">
                {Array.from({ length: 25 }).map((_, i) => (
                  <LoadingThread key={`loading-thread-${i}`} />
                ))}
              </div>
            ) : (
              <ThreadsList groupedThreads={groupedThreads} />
            )}
          </div>
        </div>
      </div>
      {/* Mobile */}
      <span className="md:hidden flex flex-row gap-2 mt-2 ml-2">
        <Sheet>
          <SheetTrigger asChild>
            <TooltipIconButton
              tooltip="New chat"
              variant="ghost"
              className="w-fit h-fit p-2"
            >
              <History className="w-6 h-6" />
            </TooltipIconButton>
          </SheetTrigger>
          <SheetContent side="left" className="bg-[#282828] border-none">
            {props.isUserThreadsLoading && !props.userThreads.length ? (
              <div className="flex flex-col gap-1 px-3 pt-3">
                {Array.from({ length: 25 }).map((_, i) => (
                  <LoadingThread key={`loading-thread-${i}`} />
                ))}
              </div>
            ) : (
              <ThreadsList groupedThreads={groupedThreads} />
            )}
          </SheetContent>
        </Sheet>
        {props.userId ? (
          <TooltipIconButton
            tooltip="New chat"
            variant="ghost"
            className="w-fit h-fit p-2"
            onClick={createThread}
          >
            <SquarePen className="w-6 h-6" />
          </TooltipIconButton>
        ) : null}
      </span>
    </span>
  );
}
