"use client"

import { Fragment, memo, useEffect, useMemo, useRef, useState } from "react"
import {
  ArticleRegularIcon,
  ChatsFillIcon,
  SidebarSimpleRightRegularIcon,
  MagnifyingGlassRegularIcon,
  XIcon,
} from "@langchain/macaw-components/icons"
import { TrashIcon } from "@phosphor-icons/react/dist/ssr/Trash"
import { Button } from "@langchain/macaw-components/Button"
import {
  ChatConversationList,
  ChatConversationListItem,
} from "@langchain/macaw-components/ChatConversationList"
import { Dialog, DialogContent } from "@langchain/macaw-components/Dialog"
import { DropdownMenuItem } from "@langchain/macaw-components/DropdownMenu"
import { EmptyState } from "@langchain/macaw-components/EmptyState"
import { IconButton } from "@langchain/macaw-components/IconButton"
import { Input } from "@langchain/macaw-components/Input"
import { Link } from "@langchain/macaw-components/Link"
import { Logo } from "@langchain/macaw-components/Logo"
import { Text } from "@langchain/macaw-components/Text"
import type { Thread } from "@/lib/hooks/threads"

interface SidebarProps {
  isCollapsed: boolean
  onToggle: () => void
  threads: Thread[]
  currentThreadId: string
  onSelectThread: (threadId: string) => void
  onDeleteThread: (threadId: string) => void
  isLoading?: boolean
  mobileOpen?: boolean
  onMobileOpenChange?: (open: boolean) => void
}

const resources = [
  {
    href: "https://smith.langchain.com",
    title: "LangSmith",
    description: "Monitoring & Tracing",
    icon: <Logo brand="langsmith" variant="logomark" className="size-3" />,
  },
  {
    href: "https://forum.langchain.com/",
    title: "Community Forum",
    description: "Join the Discussion",
    icon: <ChatsFillIcon className="size-3" />,
  },
  {
    href: "https://docs.langchain.com/oss/python/langchain/overview",
    title: "Documentation",
    description: "LangChain Docs",
    icon: <ArticleRegularIcon className="size-3" />,
  },
]

function groupThreads(threads: Thread[]) {
  const now = Date.now()
  const groups: { title: string; threads: Thread[] }[] = [
    { title: "Today", threads: [] },
    { title: "Yesterday", threads: [] },
    { title: "Previous 7 Days", threads: [] },
    { title: "Older", threads: [] },
  ]
  for (const thread of threads) {
    const age = now - new Date(thread.updated_at || thread.created_at).getTime()
    const group =
      age < 86_400_000 ? 0 : age < 172_800_000 ? 1 : age < 604_800_000 ? 2 : 3
    groups[group].threads.push(thread)
  }
  return groups.filter((group) => group.threads.length > 0)
}

export const Sidebar = memo(function Sidebar({
  isCollapsed,
  onToggle,
  threads,
  currentThreadId,
  onSelectThread,
  onDeleteThread,
  isLoading = false,
  mobileOpen = false,
  onMobileOpenChange,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [pendingDelete, setPendingDelete] = useState<Thread | null>(null)
  const desktopSearchRef = useRef<HTMLInputElement>(null)
  const mobileSearchRef = useRef<HTMLInputElement>(null)
  const mobileReturnFocusRef = useRef<HTMLElement | null>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const sidebarHasFocus = useRef(false)
  const previousCollapsed = useRef(isCollapsed)
  useEffect(() => {
    if (previousCollapsed.current !== isCollapsed && sidebarHasFocus.current)
      toggleRef.current?.focus()
    previousCollapsed.current = isCollapsed
  }, [isCollapsed])

  const filteredThreads = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    return query
      ? threads.filter(
          (thread) =>
            thread.metadata?.title?.toLowerCase().includes(query) ||
            thread.metadata?.lastMessage?.toLowerCase().includes(query)
        )
      : threads
  }, [threads, searchQuery])
  const groups = useMemo(() => groupThreads(filteredThreads), [filteredThreads])

  const toggleSidebar = () => {
    onToggle()
    requestAnimationFrame(() => toggleRef.current?.focus())
  }

  const search = (mobile: boolean) => (
    <Input
      ref={mobile ? mobileSearchRef : desktopSearchRef}
      aria-label="Search conversations"
      placeholder="Search"
      size="md"
      leftIcon={MagnifyingGlassRegularIcon}
      value={searchQuery}
      onChange={setSearchQuery}
      debounceMs={0}
      rightAction={
        searchQuery
          ? {
              icon: XIcon,
              label: "Clear search",
              onClick: () => setSearchQuery(""),
            }
          : undefined
      }
    />
  )

  const navigation = (mobile: boolean) => (
    <>
      {mobile && <div className="px-3 pb-2">{search(true)}</div>}
      <nav
        aria-label={
          mobile ? "Mobile conversation history" : "Conversation history"
        }
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
      >
        {!isLoading && searchQuery.trim() && filteredThreads.length === 0 ? (
          <div role="status">
            <EmptyState
              size="sm"
              title="No conversations found"
              description="Try a different search term."
              action={
                <Button variant="plain" onClick={() => setSearchQuery("")}>
                  Clear search
                </Button>
              }
            />
          </div>
        ) : (
          <ChatConversationList
            aria-label="Conversations"
            loading={isLoading}
            className="px-3 pt-0"
          >
            {groups.map((group, index) => (
              <Fragment key={group.title}>
                {/* Date buckets are headings, not collapsible folders. One list
                    keeps arrow-key navigation running across them. */}
                <div
                  role="listitem"
                  className={`px-3.5 pb-1 ${index === 0 ? "pt-1" : "pt-4"}`}
                >
                  <Text as="span" variant="xs" weight="medium" color="tertiary">
                    {group.title}
                  </Text>
                </div>
                {group.threads.map((thread) => (
                  <ChatConversationListItem
                    key={thread.thread_id}
                    title={thread.metadata?.title || "New conversation"}
                    selected={thread.thread_id === currentThreadId}
                    onClick={() => {
                      onSelectThread(thread.thread_id)
                      if (mobile) onMobileOpenChange?.(false)
                    }}
                    actions={{
                      items: (
                        <DropdownMenuItem
                          onSelect={() => setPendingDelete(thread)}
                          className="text-error-primary"
                        >
                          <TrashIcon
                            className="mr-2 size-4"
                            aria-hidden="true"
                          />
                          Delete conversation
                        </DropdownMenuItem>
                      ),
                    }}
                  />
                ))}
              </Fragment>
            ))}
          </ChatConversationList>
        )}
      </nav>
      <div className="flex shrink-0 flex-col justify-end px-3 py-3">
        {resources.map(({ href, title, description, icon }) => (
          <Link
            key={href}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="flex w-full rounded-lg px-3 py-1.5 hover:bg-surface-level-2 [&>span]:w-full [&>span]:no-underline"
          >
            <span className="flex items-center gap-2">
              <span
                aria-hidden="true"
                className="flex size-6 shrink-0 items-center justify-center rounded-full bg-langchain-brand/15 text-langchain-brand"
              >
                {icon}
              </span>
              <span className="flex min-w-0 flex-col">
                <Text as="span" variant="sm" weight="medium" color="primary">
                  {title}
                </Text>
                <Text as="span" variant="xs" color="quaternary">
                  {description}
                </Text>
              </span>
            </span>
          </Link>
        ))}
      </div>
    </>
  )

  // Paired with the sidebar width: same easing, same per-direction duration.
  const fade = `transition-opacity ease-[cubic-bezier(0.65,0,0.35,1)] motion-reduce:transition-none ${isCollapsed ? "opacity-0 duration-[160ms]" : "opacity-100 duration-200"}`

  return (
    <>
      <aside
        aria-label="Conversation sidebar"
        onFocusCapture={() => {
          sidebarHasFocus.current = true
        }}
        onBlurCapture={(event) => {
          if (
            event.relatedTarget &&
            !event.currentTarget.contains(event.relatedTarget as Node)
          )
            sidebarHasFocus.current = false
        }}
        // Width pushes the transcript, so it animates in place: ease-in-out,
        // faster when collapsing. Contents keep their expanded width and fade
        // so text never rewraps mid-animation.
        className={`hidden shrink-0 flex-col overflow-hidden border-r border-subtle bg-surface-level-1 transition-[width] ease-[cubic-bezier(0.65,0,0.35,1)] motion-reduce:transition-none md:flex ${isCollapsed ? "w-17 duration-[160ms]" : "w-64 duration-200"}`}
      >
        <div className="flex min-h-0 w-64 flex-1 flex-col">
          {/* 18px puts the toggle glyph on the list's text edge (12px inset
              + 14px row padding - 8px icon inset); the 68px rail keeps it
              centered when collapsed, so it never moves. */}
          <div className="flex shrink-0 items-center gap-2 pb-2 pl-4.5 pr-3 pt-3">
            <IconButton
              ref={toggleRef}
              icon={SidebarSimpleRightRegularIcon}
              label={
                isCollapsed
                  ? "Expand conversation sidebar"
                  : "Collapse conversation sidebar"
              }
              onClick={toggleSidebar}
              variant="plain"
              color="secondary"
              size="md"
            />
            <div inert={isCollapsed} className={`min-w-0 flex-1 ${fade}`}>
              {search(false)}
            </div>
          </div>
          <div
            inert={isCollapsed}
            className={`flex min-h-0 flex-1 flex-col ${fade}`}
          >
            {navigation(false)}
          </div>
        </div>
      </aside>
      <Dialog
        open={mobileOpen}
        onOpenChange={(open) => onMobileOpenChange?.(open)}
      >
        <DialogContent
          title="Conversations"
          description="Search and open your conversation history."
          className="w-[calc(100vw-2rem)] max-w-md"
          childrenClassName="flex min-h-0 flex-col p-0"
          onOpenAutoFocus={() => {
            mobileReturnFocusRef.current =
              document.activeElement instanceof HTMLElement
                ? document.activeElement
                : null
          }}
          onCloseAutoFocus={(event) => {
            if (mobileReturnFocusRef.current?.isConnected) {
              event.preventDefault()
              mobileReturnFocusRef.current.focus()
            }
          }}
        >
          <div className="flex h-[65dvh] min-h-0 flex-col">
            {navigation(true)}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null)
        }}
      >
        <DialogContent
          title="Delete conversation?"
          description={`“${pendingDelete?.metadata?.title || "New conversation"}” will be removed from your history.`}
          className="w-[calc(100vw-2rem)] max-w-md"
          onCloseAutoFocus={(event) => {
            event.preventDefault()
            requestAnimationFrame(() =>
              (
                (mobileOpen ? mobileSearchRef : desktopSearchRef).current ??
                toggleRef.current
              )?.focus()
            )
          }}
        >
          <div className="flex justify-end gap-2">
            <Button variant="outlined" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              color="error"
              onClick={() => {
                if (pendingDelete) onDeleteThread(pendingDelete.thread_id)
                setPendingDelete(null)
              }}
            >
              Delete conversation
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
})
