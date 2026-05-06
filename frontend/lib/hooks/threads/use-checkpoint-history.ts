/**
 * Checkpoint History Hook
 *
 * This hook manages fetching and displaying the checkpoint history for a LangGraph thread.
 * It provides access to all saved states (checkpoints) throughout the workflow execution,
 * allowing for state inspection, time-travel debugging, and workflow replay.
 *
 * Key Features:
 * - Fetches complete state history from LangGraph SDK
 * - Automatic refetching when thread ID changes
 * - Error handling and loading states
 * - Manual refetch capability
 */

import { useState, useEffect } from "react"
import { Client } from "@langchain/langgraph-sdk"

// ============================================================================
// Types
// ============================================================================

/**
 * Metadata associated with a checkpoint.
 * Contains execution context and step information.
 */
export interface CheckpointMetadata {
  checkpoint_id?: string
  checkpoint_ns?: string
  step?: number
  source?: string
  writes?: Record<string, any>
  created_at?: string
  parent_checkpoint_id?: string
}

/**
 * Complete checkpoint representation.
 * Represents a saved state in the workflow execution history.
 */
export interface Checkpoint {
  config?: {
    configurable?: {
      thread_id?: string
      checkpoint_id?: string
      checkpoint_ns?: string
    }
  }
  metadata?: CheckpointMetadata
  values: Record<string, any>
  next: string[]
  created_at: string
  parent_config?: {
    configurable?: {
      thread_id?: string
      checkpoint_id?: string
      checkpoint_ns?: string
    }
  }
  tasks?: Array<any>
}

/**
 * Props for the useCheckpointHistory hook.
 */
interface UseCheckpointHistoryProps {
  client: Client
  threadId: string
}

/**
 * Return type for the useCheckpointHistory hook.
 */
interface UseCheckpointHistoryReturn {
  checkpoints: Checkpoint[]
  isLoading: boolean
  error: Error | null
  refetch: () => Promise<void>
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook to fetch and manage checkpoint history for a LangGraph thread.
 *
 * Automatically fetches checkpoint history when the thread ID changes.
 * Provides loading states, error handling, and manual refetch capability.
 *
 * @param client - LangGraph SDK client instance
 * @param threadId - ID of the thread to fetch history for
 * @returns Object containing checkpoints, loading state, error state, and refetch function
 *
 * @example
 * ```tsx
 * const { checkpoints, isLoading, error, refetch } = useCheckpointHistory({
 *   client: langGraphClient,
 *   threadId: "thread-123"
 * })
 * ```
 */
export function useCheckpointHistory({
  client,
  threadId,
}: UseCheckpointHistoryProps): UseCheckpointHistoryReturn {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  /**
   * Fetches checkpoint history from the LangGraph SDK.
   * Handles API calls, error states, and data transformation.
   */
  const fetchCheckpointHistory = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const history: Checkpoint[] = []
      const stateHistory = await client.threads.getHistory(threadId)

      let checkpointIndex = 0
      for await (const state of stateHistory) {
        const stateAny = state as any

        // Debug logging for first checkpoint
        if (checkpointIndex === 0) {
          console.log("First checkpoint structure:", {
            hasConfig: !!stateAny.config,
            hasMetadata: !!stateAny.metadata,
            configKeys: stateAny.config ? Object.keys(stateAny.config) : [],
            metadataKeys: stateAny.metadata
              ? Object.keys(stateAny.metadata)
              : [],
            checkpointId:
              stateAny.config?.configurable?.checkpoint_id ||
              stateAny.metadata?.checkpoint_id,
          })
        }

        // Skip invalid checkpoints
        if (!stateAny.metadata && !stateAny.config) {
          console.warn("Skipping checkpoint with no metadata or config")
          continue
        }

        history.push({
          config: stateAny.config,
          metadata: stateAny.metadata,
          values: stateAny.values || state.values,
          next: stateAny.next || state.next || [],
          created_at:
            stateAny.created_at ||
            state.created_at ||
            new Date().toISOString(),
          parent_config: stateAny.parent_config,
          tasks: stateAny.tasks,
        })

        checkpointIndex++
      }

      console.log(
        `SUCCESS: Fetched ${history.length} checkpoints for thread ${threadId}`
      )
      setCheckpoints(history)
    } catch (err) {
      console.error("Error fetching checkpoint history:", err)
      setError(
        err instanceof Error
          ? err
          : new Error("Failed to fetch checkpoint history")
      )
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch history when thread ID changes
  useEffect(() => {
    if (threadId) {
      fetchCheckpointHistory()
    }
  }, [threadId])

  return {
    checkpoints,
    isLoading,
    error,
    refetch: fetchCheckpointHistory,
  }
}
