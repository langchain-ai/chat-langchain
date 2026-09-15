/**
 * Thread Management Hook
 *
 * Custom React hook for managing conversation threads with LangGraph backend.
 * - Fetches threads filtered by user ID
 * - Supports CRUD operations (create, read, update, delete)
 * - Implements optimistic updates for better UX
 * - Auto-filters empty threads (except the most recent)
 */

'use client'

import { useState, useEffect } from "react"
import { logger } from "../../utils/logger"
import { createLangGraphClient } from "../../api/langgraph-client"
import type { AuthRegion } from "../../auth"

// ============================================================================
// Constants
// ============================================================================

import { THREAD_FETCH_LIMIT } from "../../constants/features"

function hasThreadState(thread: { values?: Record<string, any> }): boolean {
  return Boolean(thread.values && Object.keys(thread.values).length > 0)
}

function getErrorStatus(error: unknown): number | null {
  if (!error || typeof error !== "object") return null
  const candidates = [
    (error as { status?: unknown }).status,
    (error as { statusCode?: unknown }).statusCode,
    (error as { response?: { status?: unknown } }).response?.status,
    (error as { cause?: { status?: unknown } }).cause?.status,
  ]

  for (const candidate of candidates) {
    if (typeof candidate === "number") return candidate
    if (typeof candidate === "string") {
      const parsed = Number(candidate)
      if (Number.isFinite(parsed)) return parsed
    }
  }

  return null
}

function isHttpStatusError(error: unknown, status: number): boolean {
  if (getErrorStatus(error) === status) return true
  return error instanceof Error && error.message.includes(String(status))
}

/**
 * Statuses that mean "this one thread is not readable by the current actor":
 * deleted (404), or owned by a different actor (403, which is what per-actor
 * thread scoping returns after a guest identity rotates). Neither ever means
 * the user has no history.
 */
const UNREACHABLE_THREAD_STATUSES = [403, 404]

function classifyThreadFetchError(
  error: unknown
): "prunable" | "unreachable" | "failed" {
  const status = getErrorStatus(error)
  if (status !== null) {
    return UNREACHABLE_THREAD_STATUSES.includes(status) ? "prunable" : "failed"
  }

  // No numeric status, so fall back to the message - but treat that as
  // low-confidence. Thread ids are UUIDs and routinely contain digit runs like
  // "404", so a match here hides the thread without forgetting its id.
  const mentionsStatus =
    error instanceof Error &&
    UNREACHABLE_THREAD_STATUSES.some((candidate) =>
      error.message.includes(String(candidate))
    )
  return mentionsStatus ? "unreachable" : "failed"
}

// ============================================================================
// Types
// ============================================================================

/**
 * Client profile information for user identification and display.
 */
export interface ClientProfile {
  id: string
  label?: string
  avatarColor?: string
}

/**
 * Thread metadata stored with each conversation.
 * Contains user info, title, and custom fields.
 */
export interface ThreadMetadata {
  user_id: string
  title?: string
  lastMessage?: string
  client?: ClientProfile
  [key: string]: any // Allow additional metadata fields
}

/**
 * Thread object representing a conversation.
 * Extends LangGraph's base Thread type with our metadata.
 */
export interface Thread {
  thread_id: string
  created_at: string
  updated_at: string
  metadata: ThreadMetadata
  values?: Record<string, any>
}

/**
 * Result of resolving one stored thread id.
 * - `ok`: the thread loaded.
 * - `unreachable`: authoritatively not readable; `prunable` marks the cases
 *   confident enough to forget the id locally.
 * - `failed`: anything else (network, 5xx, auth outage) - never interpreted as
 *   an absence of history.
 */
type ThreadFetchOutcome =
  | { kind: "ok"; thread: Thread }
  | { kind: "unreachable"; threadId: string; prunable: boolean }
  | { kind: "failed"; threadId: string; error: unknown }

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook to manage conversation threads.
 *
 * Features:
 * - Auto-loads threads when userId is available
 * - Filters out empty threads (except most recent)
 * - Optimistic updates for instant UI feedback
 * - Server-side storage via LangGraph
 *
 * @param userId - The current user's thread owner ID.
 * @param authToken - Credential used to authenticate LangGraph requests.
 * @returns Thread management functions and state
 */
interface UseThreadsOptions {
  threadIds?: string[]
  authRegion?: AuthRegion
  /**
   * Called with thread ids the backend confirms this identity cannot read, so
   * the caller can stop re-requesting them on every load.
   */
  onUnreachableThreadIds?: (threadIds: string[]) => void
}

export function useThreads(
  userId: string | undefined,
  authToken?: string,
  options: UseThreadsOptions = {}
) {
  const [isLoading, setIsLoading] = useState(false)
  const [threads, setThreads] = useState<Thread[]>([])
  const threadIdsKey = options.threadIds?.join(",") ?? null
  const authRegion = options.authRegion

  // Auto-load threads when user ID becomes available
  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!userId || !authToken) {
      setThreads([])
      return
    }
    getUserThreads(userId)
  }, [userId, authToken, threadIdsKey, authRegion])

  // ==========================================================================
  // Thread Fetching
  // ==========================================================================

  /**
   * Fetch all threads for a specific user from LangGraph backend.
   * Filters out empty threads except the most recent one.
   * 
   * @param id - User ID to fetch threads for
   * @param silent - If true, skip setting loading state (for background refetches)
   */
  const getUserThreads = async (id: string, silent = false): Promise<void> => {
    if (!silent) {
      setIsLoading(true)
    }
    try {
      const client = createLangGraphClient(authToken, authRegion)

      logger.info('Fetching threads for user:', id)

      let userThreads: Thread[]
      if (options.threadIds) {
        const outcomes: ThreadFetchOutcome[] = await Promise.all(
          options.threadIds.map(async (threadId): Promise<ThreadFetchOutcome> => {
            try {
              const thread = (await client.threads.get(threadId)) as any as Thread
              return { kind: "ok", thread }
            } catch (error) {
              const classification = classifyThreadFetchError(error)
              if (classification === "failed") {
                return { kind: "failed", threadId, error }
              }
              // 403 here means the thread belongs to a different actor - the
              // signal that this guest identity rotated - while 404 means it is
              // genuinely gone. Worth telling apart in the field.
              logger.debug(
                `Thread ${threadId} not readable by this identity (status ${
                  getErrorStatus(error) ?? "unknown"
                })`
              )
              return {
                kind: "unreachable",
                threadId,
                prunable: classification === "prunable",
              }
            }
          })
        )

        const failures = outcomes.filter(
          (outcome): outcome is Extract<ThreadFetchOutcome, { kind: "failed" }> =>
            outcome.kind === "failed"
        )
        if (failures.length > 0) {
          // A transient failure is not evidence the user has no history, so
          // leave whatever is already on screen in place.
          logger.error('Error fetching threads:', failures[0].error)
          return
        }

        const prunable = outcomes
          .filter(
            (outcome): outcome is Extract<ThreadFetchOutcome, { kind: "unreachable" }> =>
              outcome.kind === "unreachable"
          )
          .filter((outcome) => outcome.prunable)
          .map((outcome) => outcome.threadId)
        if (prunable.length > 0) {
          logger.info(
            `Forgetting ${prunable.length} stored thread id(s) this identity cannot read`
          )
          options.onUnreachableThreadIds?.(prunable)
        }

        userThreads = outcomes
          .filter(
            (outcome): outcome is Extract<ThreadFetchOutcome, { kind: "ok" }> =>
              outcome.kind === "ok"
          )
          .map((outcome) => outcome.thread)
      } else {
        userThreads = (await client.threads.search({
          metadata: {
            user_id: id,
          },
          limit: THREAD_FETCH_LIMIT,
        })) as any[] as Thread[]
      }

      logger.info('Fetched threads:', userThreads.length)

      setThreads(userThreads.filter(hasThreadState))
    } catch (error) {
      // Preserve the current list rather than blanking the sidebar on failure.
      logger.error('Error fetching threads:', error)
    } finally {
      if (!silent) {
        setIsLoading(false)
      }
    }
  }

  /**
   * Get a specific thread by ID.
   *
   * @param id - Thread ID to fetch
   * @returns Thread object or null if not found
   */
  const getThreadById = async (id: string): Promise<Thread | null> => {
    try {
      const client = createLangGraphClient(authToken, authRegion)
      const thread = (await client.threads.get(id)) as any as Thread
      return thread
    } catch (error) {
      // Silently handle 404 errors (thread doesn't exist or was deleted)
      if (isHttpStatusError(error, 404)) {
        logger.debug(`Thread ${id} not found (404)`)
        return null
      }
      logger.error('Error fetching thread:', error)
      return null
    }
  }

  // ==========================================================================
  // Thread Updates
  // ==========================================================================

  /**
   * Update thread metadata (title, last message, etc.).
   * Uses optimistic updates for instant UI feedback.
   *
   * @param threadId - ID of thread to update
   * @param metadata - Partial metadata to merge with existing
   */
  const updateThreadMetadata = async (
    threadId: string,
    metadata: Partial<ThreadMetadata>
  ): Promise<void> => {
    try {
      // Optimistically update the UI immediately
      setThreads((prev) => {
        const existingThread = prev.find(t => t.thread_id === threadId)

        if (existingThread) {
          // Update existing thread
          return prev.map(t =>
            t.thread_id === threadId
              ? {
                  ...t,
                  updated_at: new Date().toISOString(),
                  metadata: {
                    ...t.metadata,
                    ...metadata,
                  }
                }
              : t
          )
        } else {
          // Thread doesn't exist yet, return unchanged
          return prev
        }
      })

      const client = createLangGraphClient(authToken, authRegion)

      // Get current thread to merge metadata
      const currentThread = await client.threads.get(threadId) as any as Thread
      const currentMetadata = (currentThread.metadata || {}) as ThreadMetadata

      // Update thread with merged metadata in background
      await client.threads.update(threadId, {
        metadata: {
          ...currentMetadata,
          ...metadata,
        },
      })

      // Don't refetch - the optimistic update already handled the UI
    } catch (error) {
      // Silently handle 404 errors (thread doesn't exist or was deleted)
      if (isHttpStatusError(error, 404)) {
        logger.debug(`Thread ${threadId} not found for metadata update (404)`)
        return
      }
      logger.error('Error updating thread metadata:', error)
      // On error, refetch to get correct state
      if (metadata.user_id) {
        await getUserThreads(metadata.user_id)
      }
    }
  }

  /**
   * Optimistically add a thread to the UI before backend confirmation.
   * Useful for instant feedback when creating new threads.
   *
   * @param thread - Thread to add to the list
   */
  const addOptimisticThread = (thread: Thread): void => {
    setThreads((prev) => {
      // Check if thread already exists
      const exists = prev.some(t => t.thread_id === thread.thread_id)
      if (exists) return prev

      // Add at the beginning
      return [thread, ...prev]
    })
  }

  // ==========================================================================
  // Thread Deletion
  // ==========================================================================

  /**
   * Delete a thread from the backend.
   * Uses optimistic updates and provides callback for current thread cleanup.
   *
   * @param id - Thread ID to delete
   * @param onDeleteCurrent - Optional callback if deleting current thread
   */
  const deleteThread = async (
    id: string,
    onDeleteCurrent?: () => void
  ): Promise<void> => {
    if (!userId || !authToken) {
      throw new Error("User ID not found")
    }

    // Optimistically update UI
    setThreads((prevThreads) => {
      const newThreads = prevThreads.filter(
        (thread) => thread.thread_id !== id,
      )
      return newThreads
    })

    try {
      const client = createLangGraphClient(authToken, authRegion)
      await client.threads.delete(id)
      logger.info('Thread deleted:', id)

      // If callback provided (e.g., to clear current thread), invoke it
      if (onDeleteCurrent) {
        onDeleteCurrent()
      }

      // Refetch silently to ensure consistency (no loader flash since we already optimistically updated)
      await getUserThreads(userId, true)
    } catch (error) {
      // Silently handle 404 errors (thread already deleted)
      if (isHttpStatusError(error, 404)) {
        logger.debug(`Thread ${id} already deleted (404)`)
        return
      }
      logger.error('Error deleting thread:', error)
      // Revert optimistic update on error (show loader since something went wrong)
      await getUserThreads(userId)
    }
  }

  // ==========================================================================
  // Return API
  // ==========================================================================

  return {
    isLoading,
    threads,
    getThreadById,
    setThreads,
    getUserThreads,
    updateThreadMetadata,
    deleteThread,
    addOptimisticThread,
  }
}
