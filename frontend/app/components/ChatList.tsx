
import { ThreadListProps } from "../hooks/useThreadList";
import { useThread } from "../hooks/useThread";

import { Text } from "@chakra-ui/react";
import { PlusSquareIcon  } from "@chakra-ui/icons"

export function ChatList(props: {
  threads: ThreadListProps["threads"];
  enterChat: (id: string | null) => void;
  deleteChat: (id: string) => void;
}) {
  const { currentThread } = useThread();

  // State for tracking which chat's menu is visible
  // const [visibleMenu, setVisibleMenu] = useState<string | null>(null);

  // Event listener to close the menu when clicking outside of it
  // useEffect(() => {
    // const closeMenu = () => setVisibleMenu(null);
    // window.addEventListener("click", closeMenu);
    // return () => window.removeEventListener("click", closeMenu);
  // }, []);

  return (
    <>
      <>
        <PlusSquareIcon color={"white"} onClick={() => props.enterChat(null)}/>
        <Text color={"white"}>New chat</Text>
      </>
      <ul role="list" className="-mx-2 mt-2 space-y-1">
        {props.threads?.map(thread => (
          <li
            key={thread.thread_id}
          >
            <div
              onClick={() => props.enterChat(thread.thread_id)}
              className={
                "group flex items-center gap-x-3 rounded-md px-2 leading-6 cursor-pointer flex-grow min-w-0"
              }
            >
              <span
                className={(
                  thread.thread_id === currentThread?.thread_id
                    ? "text-indigo-600 border-indigo-600"
                    : "text-gray-400 border-gray-200 group-hover:border-indigo-600 group-hover:text-indigo-600") + 
                  " flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border text-[0.625rem] font-medium bg-white"
                }
              >
                {thread.metadata?.name as string ?? " "}
              </span>
            </div>
            {/* Menu Dropdown
            {visibleMenu === thread.thread_id && (
              <div className="origin-top-right absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 focus:outline-none z-10">
                <div
                  className="py-1"
                  role="menu"
                  aria-orientation="vertical"
                  aria-labelledby="options-menu"
                >
                  <a
                    href="#"
                    className="text-gray-700 block px-4 py-2 text-sm hover:bg-gray-100"
                    role="menuitem"
                    onClick={(event) => {
                      event.preventDefault();
                      if (
                        window.confirm(
                          `Are you sure you want to delete chat "${thread.name}"?`,
                        )
                      ) {
                        props.deleteChat(thread.thread_id);
                      }
                    }}
                  >
                    Delete
                  </a>
                </div>
              </div>
            )} */}
          </li>
        )) ?? (
          <li className="leading-6 p-2 animate-pulse font-black text-gray-400 text-lg">
            ...
          </li>
        )}
      </ul>
    </>
  );
}