import { Thread, ThreadProps } from "./thread-item";
import { prettifyDateLabel } from "./utils";

export interface ThreadsListProps {
  groupedThreads: {
    today: ThreadProps[];
    yesterday: ThreadProps[];
    lastSevenDays: ThreadProps[];
    older: ThreadProps[];
  };
}

export function ThreadsList(props: ThreadsListProps) {
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
