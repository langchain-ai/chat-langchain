import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { SquarePen, History } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "../ui/sheet";
import { useToast } from "../../hooks/use-toast";
import { Skeleton } from "../ui/skeleton";
import React from "react";
import { useGraphContext } from "../../contexts/GraphContext";
import { groupThreads } from "./utils";
import { ThreadsList } from "./thread-list";

const LoadingThread = () => <Skeleton className="w-full h-8 bg-[#373737]" />;

function ThreadHistoryComponent() {
  const { toast } = useToast();
  const { threadsData, userData, graphData } = useGraphContext();
  const {
    userThreads,
    isUserThreadsLoading,
    threadId,
    deleteThread,
    createThread,
    getUserThreads,
  } = threadsData;
  const { userId } = userData;
  const { messages, switchSelectedThread, setMessages } = graphData;
  const isEmpty = messages.length === 0;

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

  const createAndSetupNewThread = async () => {
    if (!userId) {
      toast({
        title: "Error creating thread",
        description: "Your user ID was not found. Please try again later.",
      });
      return;
    }
    const currentThread = userThreads.find(
      (thread) => thread.thread_id === threadId,
    );
    if (currentThread && !currentThread.values && isEmpty) {
      return;
    }

    clearMessages();
    await createThread(userId);
    // Re-fetch threads so that the new thread shows up.
    await getUserThreads(userId);
  };

  return (
    <span>
      {/* Tablet & up */}
      <div className="hidden md:flex flex-col w-[260px] h-full">
        <div className="flex-grow border-r-[1px] border-[#393939] my-6 flex flex-col overflow-hidden">
          <div className="flex flex-row items-center justify-between border-b-[1px] border-[#393939] pt-3 px-2 mx-4 -mt-4 text-gray-200">
            <p className="text-lg font-medium">Chat History</p>
            {userId ? (
              <TooltipIconButton
                tooltip="New chat"
                variant="ghost"
                className="w-fit p-2"
                onClick={createAndSetupNewThread}
              >
                <SquarePen className="w-5 h-5" />
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
            onClick={createAndSetupNewThread}
          >
            <SquarePen className="w-6 h-6" />
          </TooltipIconButton>
        ) : null}
      </span>
    </span>
  );
}

export const ThreadHistory = React.memo(ThreadHistoryComponent);
