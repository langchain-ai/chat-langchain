import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { SquarePen, History } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "../ui/sheet";
import { Skeleton } from "../ui/skeleton";
import React from "react";
import { useGraphContext } from "../../contexts/GraphContext";
import { groupThreads } from "./utils";
import { ThreadsList } from "./thread-list";
import { useQueryState } from "nuqs";

const LoadingThread = () => <Skeleton className="w-full h-8 bg-[#373737]" />;

function ThreadHistoryComponent() {
  const { threadsData, userData, graphData } = useGraphContext();
  const { userThreads, isUserThreadsLoading, deleteThread } = threadsData;
  const { userId } = userData;
  const { switchSelectedThread, setMessages } = graphData;
  const [_threadId, setThreadId] = useQueryState("threadId");

  const clearMessages = () => {
    setMessages([]);
  };

  const deleteThreadAndClearMessages = async (id: string) => {
    clearMessages();
    await deleteThread(id, clearMessages);
  };

  const groupedThreads = groupThreads(
    userThreads,
    switchSelectedThread,
    deleteThreadAndClearMessages,
  );

  const createNewSession = async () => {
    setThreadId(null);
    clearMessages();
  };

  return (
    <div className="h-full">
      {/* Tablet & up */}
      <div className="hidden lg:flex flex-col w-[260px] h-full bg-[#F9F9F9]">
        <div className="flex-grow my-6 flex flex-col overflow-hidden">
          <div className="flex flex-row items-center justify-between border-b-[1px] pt-3 px-2 mx-4 -mt-4 text-gray-200">
            <p className="text-lg text-black font-bold">Chat History</p>
            {userId ? (
              <TooltipIconButton
                tooltip="New chat"
                variant="ghost"
                className="w-fit p-2"
                onClick={createNewSession}
              >
                <SquarePen className="w-5 h-5 text-black" />
              </TooltipIconButton>
            ) : null}
          </div>
          <div className="overflow-y-auto flex-grow scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
            {isUserThreadsLoading && !userThreads.length ? (
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
      <span className="lg:hidden flex flex-row m-4">
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
            {isUserThreadsLoading && !userThreads.length ? (
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
        {userId ? (
          <TooltipIconButton
            tooltip="New chat"
            variant="ghost"
            className="w-fit h-fit p-2"
            onClick={createNewSession}
          >
            <SquarePen className="w-6 h-6" />
          </TooltipIconButton>
        ) : null}
      </span>
    </div>
  );
}

export const ThreadHistory = React.memo(ThreadHistoryComponent);
