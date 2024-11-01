import { Thread } from "@langchain/langgraph-sdk";
import { ThreadProps } from "./thread-item";
import { subDays, isToday, isYesterday, isWithinInterval } from "date-fns";

export function convertThreadActualToThreadProps(
  thread: Thread,
  switchSelectedThread: (thread: Thread) => void,
  deleteThread: (id: string) => void,
): ThreadProps {
  const values = thread.values as Record<string, any> | undefined;
  return {
    id: thread.thread_id,
    label: values?.messages?.[0].content || "Untitled",
    createdAt: new Date(thread.created_at),
    onClick: () => {
      return switchSelectedThread(thread);
    },
    onDelete: () => {
      return deleteThread(thread.thread_id);
    },
  };
}

export function groupThreads(
  threads: Thread[],
  switchSelectedThread: (thread: Thread) => void,
  deleteThread: (id: string) => void,
) {
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
}

export function prettifyDateLabel(group: string): string {
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
}
