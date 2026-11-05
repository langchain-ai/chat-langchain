/**
 * Chat State Hook
 *
 * Manages UI state for the chat interface using useReducer for better state organization.
 * Consolidates multiple useState hooks into a single, predictable state management system.
 */

import { useReducer, useCallback } from "react"
import { MAX_INPUT_CHARS, STORAGE_KEYS } from "../../constants/features"

import { chatUIReducer } from "../../utils/chat/composer-state"
export type { ChatUIState, ChatUIAction } from "../../utils/chat/composer-state"

// ============================================================================
// Hook
// ============================================================================

/**
 * Hook to manage chat UI state with localStorage persistence for input.
 *
 * @param threadId - Current thread ID for draft persistence
 * @returns Chat UI state and dispatch function
 *
 * @example
 * ```tsx
 * const { state, dispatch, setInput } = useChatState(threadId)
 *
 * // Update input
 * setInput("Hello world")
 *
 * // Start sending
 * dispatch({ type: 'START_SEND' })
 *
 * // Finish sending
 * dispatch({ type: 'FINISH_SEND' })
 * ```
 */
export function useChatState(threadId: string) {
  // Initialize input from localStorage if available
  const initialInput = typeof window !== 'undefined'
    ? (localStorage.getItem(`${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`) || '').slice(0, MAX_INPUT_CHARS)
    : ''

  const [state, dispatch] = useReducer(chatUIReducer, {
    input: initialInput,
    copiedId: null,
    isLoading: false,
    isLoadingThread: false,
    isRegenerating: false,
    isStopping: false,
    hasAutoSent: false,
    errorMessage: null,
  })

  /**
   * Set input with localStorage persistence.
   */
  const setInput = useCallback((value: string) => {
    const cappedValue = value.slice(0, MAX_INPUT_CHARS)
    dispatch({ type: 'SET_INPUT', payload: cappedValue })

    // Auto-save draft to localStorage
    if (typeof window !== 'undefined') {
      if (cappedValue) {
        localStorage.setItem(`${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`, cappedValue)
      } else {
        localStorage.removeItem(`${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`)
      }
    }
  }, [threadId])

  /**
   * Clear input and remove draft from localStorage.
   */
  const clearInput = useCallback(() => {
    dispatch({ type: 'RESET_INPUT' })
    if (typeof window !== 'undefined') {
      localStorage.removeItem(`${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`)
    }
  }, [threadId])

  return {
    state,
    dispatch,
    setInput,
    clearInput,
  }
}
