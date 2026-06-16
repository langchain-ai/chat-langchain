"use client"

import { Suspense, useState, useEffect, useMemo, useRef } from "react"
import { useQueryState } from "nuqs"
import { Sidebar } from "@/components/layout/sidebar"
import { Header } from "@/components/layout/header"
import { ChatInterface } from "@/components/chat/chat-interface"
import { KeyboardShortcutsDialog } from "@/components/layout/keyboard-shortcuts-dialog"
import { useThreads, type ClientProfile } from "@/lib/hooks/threads"
import {
  addStoredGuestThreadId,
  getStoredGuestThreadIds,
  removeStoredGuestThreadId,
} from "@/lib/hooks/threads/guest-thread-storage"
import { useLangGraphAuth, useClientProfile } from "@/lib/hooks/auth"
import { resolveClientProfile } from "@/lib/config/client-config"
import { generateQuickTitle, generateThreadTitle } from "@/lib/utils/string"
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts"

function DashboardContent() {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)
  const [showShortcutsDialog, setShowShortcutsDialog] = useState(false)

  // Track newly created threads that haven't been initialized in backend yet
  const [newThreads, setNewThreads] = useState<Set<string>>(new Set())

  // Use URL query param for thread ID (shareable, bookmarkable)
  const [threadId, setThreadId] = useQueryState("threadId")

  // Support ?q=... for auto-sending a prompt on page load
  const [initialPrompt, setInitialPrompt] = useQueryState("q")

  // Resolve the current thread owner identity and credential.
  const {
    userId,
    authToken,
    authRegion,
    guestUserId,
    guestToken,
    loading: langGraphAuthLoading,
  } = useLangGraphAuth()
  const shouldLoadGuestThreads = Boolean(
    userId && guestUserId && guestToken && userId !== guestUserId
  )
  const isCurrentUserGuest = Boolean(userId && guestUserId && userId === guestUserId)
  const [selectedThreadOwner, setSelectedThreadOwner] = useState<{
    threadId: string
    ownerId: string | null
  } | null>(null)
  const [guestThreadIds, setGuestThreadIds] = useState<string[]>(() =>
    typeof window === "undefined" ? [] : getStoredGuestThreadIds()
  )
  const previousUserIdRef = useRef<string | null>(null)

  // Load threads from LangGraph backend
  const {
    threads: primaryThreads,
    isLoading: primaryThreadsLoading,
    updateThreadMetadata: updatePrimaryThreadMetadata,
    deleteThread: deletePrimaryThread,
    addOptimisticThread: addPrimaryOptimisticThread,
  } = useThreads(userId || undefined, authToken || undefined, {
    threadIds: isCurrentUserGuest ? guestThreadIds : undefined,
    authRegion,
  })
  const {
    threads: guestThreads,
    isLoading: guestThreadsLoading,
    updateThreadMetadata: updateGuestThreadMetadata,
    deleteThread: deleteGuestThread,
    addOptimisticThread: addGuestOptimisticThread,
  } = useThreads(
    shouldLoadGuestThreads ? guestUserId || undefined : undefined,
    shouldLoadGuestThreads ? guestToken || undefined : undefined,
    { threadIds: guestThreadIds, authRegion }
  )

  const threads = useMemo(() => {
    const byId = new Map<string, typeof primaryThreads[number]>()

    for (const thread of guestThreads) {
      byId.set(thread.thread_id, thread)
    }
    for (const thread of primaryThreads) {
      byId.set(thread.thread_id, thread)
    }

    return Array.from(byId.values()).sort((a, b) => {
      const aTime = new Date(a.updated_at || a.created_at).getTime()
      const bTime = new Date(b.updated_at || b.created_at).getTime()
      return bTime - aTime
    })
  }, [guestThreads, primaryThreads])

  const threadsLoading = primaryThreadsLoading || guestThreadsLoading

  useEffect(() => {
    if (!threadId) return

    const selectedThread = threads.find((thread) => thread.thread_id === threadId)
    const ownerId = selectedThread?.metadata?.user_id
    if (!ownerId) return

    setSelectedThreadOwner((current) => {
      if (current?.threadId === threadId && current.ownerId === ownerId) {
        return current
      }
      return { threadId, ownerId }
    })
  }, [threadId, threads])

  const currentThreadOwnerId = useMemo(() => {
    if (selectedThreadOwner?.threadId === threadId) {
      return selectedThreadOwner.ownerId
    }

    const selectedThread = threads.find((thread) => thread.thread_id === threadId)
    return selectedThread?.metadata?.user_id || userId
  }, [selectedThreadOwner, threadId, threads, userId])

  const currentThreadAuthToken = useMemo(() => {
    if (
      currentThreadOwnerId &&
      guestUserId &&
      currentThreadOwnerId === guestUserId
    ) {
      return guestToken
    }
    if (currentThreadOwnerId && currentThreadOwnerId === userId) {
      return authToken
    }
    if (!currentThreadOwnerId) {
      return authToken
    }
    return null
  }, [authToken, currentThreadOwnerId, guestToken, guestUserId, userId])

  useEffect(() => {
    if (!threadId || !userId || langGraphAuthLoading || currentThreadAuthToken) {
      return
    }

    const newThreadId = crypto.randomUUID()
    setNewThreads(prev => new Set(prev).add(newThreadId))
    setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
    setThreadId(newThreadId)
  }, [currentThreadAuthToken, langGraphAuthLoading, threadId, setThreadId, userId])

  useEffect(() => {
    if (langGraphAuthLoading || !userId) {
      return
    }

    const previousUserId = previousUserIdRef.current
    previousUserIdRef.current = userId

    if (!previousUserId || previousUserId === userId || !threadId) {
      return
    }

    const newThreadId = crypto.randomUUID()
    setNewThreads(prev => new Set(prev).add(newThreadId))
    setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
    setThreadId(newThreadId)
  }, [
    langGraphAuthLoading,
    setThreadId,
    threadId,
    userId,
  ])

  const { clientProfile } = useClientProfile()

  // Create a new thread
  const handleNewChat = () => {
    const newThreadId = crypto.randomUUID()

    // Mark this thread as new (doesn't exist in backend yet)
    setNewThreads(prev => new Set(prev).add(newThreadId))

    setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
    setThreadId(newThreadId)
  }

  // Switch to an existing thread
  const handleSelectThread = (selectedThreadId: string) => {
    const selectedThread = threads.find(
      (thread) => thread.thread_id === selectedThreadId
    )
    setSelectedThreadOwner({
      threadId: selectedThreadId,
      ownerId: selectedThread?.metadata?.user_id || userId,
    })
    setThreadId(selectedThreadId)
  }

  // Delete a thread
  const handleDeleteThread = (threadIdToDelete: string) => {
    const threadOwnerId =
      threads.find((thread) => thread.thread_id === threadIdToDelete)?.metadata
        ?.user_id || userId
    const deleteForOwner =
      shouldLoadGuestThreads && threadOwnerId === guestUserId
        ? deleteGuestThread
        : deletePrimaryThread

    if (threadOwnerId === guestUserId || threadOwnerId?.startsWith("user-")) {
      setGuestThreadIds(removeStoredGuestThreadId(threadIdToDelete))
    }

    deleteForOwner(threadIdToDelete, () => {
      // If deleting current thread, create a new one
      if (threadIdToDelete === threadId) {
        const newThreadId = crypto.randomUUID()
        setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
        setThreadId(newThreadId)
      }
    })
  }

  // Handle when thread is not found (404) or access denied (403)
  const handleThreadNotFound = () => {
    console.log('Thread not accessible - creating new thread')

    // Always create a new thread when current thread is not accessible
    const newThreadId = crypto.randomUUID()

    // Mark this thread as new (doesn't exist in backend yet)
    setNewThreads(prev => new Set(prev).add(newThreadId))

    setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
    setThreadId(newThreadId)
  }

  // Update thread metadata when messages are sent
  const handleThreadUpdate = async (
    threadId: string,
    title: string,
    lastMessage: string,
    client?: ClientProfile,
    messageCount?: number, // Track how many messages are in the thread
  ) => {
    const ownerId = currentThreadOwnerId || userId
    if (!ownerId) return

    const isGuestOwnedThread =
      Boolean(guestUserId && guestUserId === ownerId) || ownerId.startsWith("user-")
    const isGuestThread =
      shouldLoadGuestThreads && isGuestOwnedThread && userId !== ownerId
    const updateThreadForOwner = isGuestThread
      ? updateGuestThreadMetadata
      : updatePrimaryThreadMetadata
    const addOptimisticThreadForOwner = isGuestThread
      ? addGuestOptimisticThread
      : addPrimaryOptimisticThread

    if (isGuestOwnedThread) {
      setGuestThreadIds(addStoredGuestThreadId(threadId))
    }

    // Clear the new thread flag once the thread has been initialized (first message sent)
    if (newThreads.has(threadId)) {
      setNewThreads(prev => {
        const updated = new Set(prev)
        updated.delete(threadId)
        return updated
      })
    }

    const resolvedClient = resolveClientProfile(client ?? clientProfile)

    // Check if this thread already exists
    const existingThread = threads.find(t => t.thread_id === threadId)
    const isUntitledThread = existingThread?.metadata?.title === "Untitled"
    const shouldGenerateAITitle = !existingThread || // First message (thread doesn't exist)
                                  isUntitledThread || // First real message (was "Untitled")
                                  (messageCount && messageCount > 1 && messageCount % 5 === 0) // Every 5 messages after

    if (!existingThread || isUntitledThread) {
      // First message: Keep "Untitled" while AI title generates, then replace directly

      if (!existingThread) {
        // Thread doesn't exist at all - add it with "Untitled"
        addOptimisticThreadForOwner({
          thread_id: threadId,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          metadata: {
            user_id: ownerId,
            title: "Untitled",
            lastMessage,
            client: resolvedClient,
          },
        })
      }

      // Update last message immediately (keep "Untitled" for now)
      await updateThreadForOwner(threadId, {
        user_id: ownerId,
        lastMessage,
        client: resolvedClient,
      })

      // Generate AI title in background - goes straight from "Untitled" to AI title
      generateThreadTitle({
        userMessage: title,
        assistantResponse: lastMessage,
      }).then((aiTitle) => {
        if (aiTitle.length > 0) {
          console.log('Setting AI title:', aiTitle)
          updateThreadForOwner(threadId, {
            user_id: ownerId,
            title: aiTitle,
            lastMessage,
            client: resolvedClient,
          })
        }
      }).catch((error) => {
        console.error('Failed to generate AI title:', error)
        // Fallback to quick title if AI fails
        const quickTitle = generateQuickTitle(title)
        updateThreadForOwner(threadId, {
          user_id: ownerId,
          title: quickTitle,
          lastMessage,
          client: resolvedClient,
        })
      })
    } else if (shouldGenerateAITitle && messageCount) {
      // Every 5 messages: Regenerate AI title based on conversation
      console.log(`Regenerating AI title at message ${messageCount}`)

      // Update last message immediately
      await updateThreadForOwner(threadId, {
        user_id: ownerId,
        lastMessage,
        client: resolvedClient,
      })

      // Generate new AI title in background
      generateThreadTitle({
        userMessage: title,
        assistantResponse: lastMessage,
      }).then((aiTitle) => {
        if (aiTitle.length > 0) {
          console.log('Updated title at message', messageCount, '→', aiTitle)
          updateThreadForOwner(threadId, {
            user_id: ownerId,
            title: aiTitle,
            lastMessage,
            client: resolvedClient,
          })
        }
      }).catch((error) => {
        console.error('Failed to regenerate AI title:', error)
      })
    } else {
      // Regular update: Just update last message, keep existing title
      await updateThreadForOwner(threadId, {
        user_id: ownerId,
        lastMessage,
        client: resolvedClient,
      })
    }
  }

  // If no threadId in URL, create one
  // Also create a new thread if ?q= is present (always start fresh for prompt links)
  const hasProcessedPromptRef = useRef(false)
  useEffect(() => {
    if (!userId) return

    // Validate and process ?q= param - create fresh thread for prompt links
    const trimmedPrompt = initialPrompt?.trim()
    if (trimmedPrompt && !hasProcessedPromptRef.current) {
      hasProcessedPromptRef.current = true
      const newThreadId = crypto.randomUUID()
      setNewThreads(prev => new Set(prev).add(newThreadId))
      setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
      setThreadId(newThreadId)
      return
    }

    // Create a thread if none exists
    if (!threadId) {
      const newThreadId = crypto.randomUUID()
      setNewThreads(prev => new Set(prev).add(newThreadId))
      setSelectedThreadOwner({ threadId: newThreadId, ownerId: userId })
      setThreadId(newThreadId)
    }
  }, [
    threadId,
    setThreadId,
    initialPrompt,
    userId,
  ])

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      shortcut: {
        key: '/',
        metaKey: true,
        description: 'Toggle keyboard shortcuts',
        category: 'Navigation',
      },
      handler: () => setShowShortcutsDialog(!showShortcutsDialog),
    },
    {
      shortcut: {
        key: 'b',
        metaKey: true,
        description: 'Toggle sidebar',
        category: 'Navigation',
      },
      handler: () => setIsSidebarCollapsed(!isSidebarCollapsed),
    },
    {
      shortcut: {
        key: 'i',
        metaKey: true,
        description: 'Create new chat',
        category: 'Navigation',
      },
      handler: handleNewChat,
    },
  ])

  return (
    <>
      <KeyboardShortcutsDialog
        open={showShortcutsDialog}
        onOpenChange={setShowShortcutsDialog}
      />
      <div className="flex h-screen bg-background">
        <Sidebar
          isCollapsed={isSidebarCollapsed}
          onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          threads={threads}
          currentThreadId={threadId || ''}
          onSelectThread={handleSelectThread}
          onDeleteThread={handleDeleteThread}
          isLoading={threadsLoading}
        />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header
          onNewChat={handleNewChat}
        />
        {threadId && (
          <ChatInterface
            key={`${threadId}:${currentThreadOwnerId || ""}`}
            showToolCalls={showToolCalls}
            threadId={threadId}
            userId={currentThreadOwnerId}
            authToken={currentThreadAuthToken}
            authRegion={authRegion}
            onThreadUpdate={handleThreadUpdate}
            onThreadNotFound={handleThreadNotFound}
            isNewThread={newThreads.has(threadId)}
            initialMessage={initialPrompt}
            autoSend={!!initialPrompt}
            onInitialMessageSent={() => setInitialPrompt(null)}
          />
        )}
      </div>
    </div>
    </>
  )
}

export default function DashboardPage() {
  return (
    <Suspense fallback={
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    }>
      <DashboardContent />
    </Suspense>
  )
}
