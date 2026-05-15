/**
 * Chat State Hook
 *
 * Manages UI state for the chat interface using useReducer for better state organization.
 * Consolidates multiple useState hooks into a single, predictable state management system.
 */

import { useReducer, useCallback } from "react"
import { MAX_INPUT_CHARS, STORAGE_KEYS } from "../../constants/features"

// ============================================================================
// Types
// ============================================================================

/**
 * UI state for the chat interface.
 */
export interface ChatUIState {
  input: string
  copiedId: string | null
  isLoading: boolean
  isLoadingThread: boolean
  isRegenerating: boolean
  isStopping: boolean
  hasAutoSent: boolean
  errorMessage: string | null
}

/**
 * Actions for updating chat state.
 */
export type ChatUIAction =
  | { type: 'SET_INPUT'; payload: string }
  | { type: 'SET_COPIED_ID'; payload: string | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_LOADING_THREAD'; payload: boolean }
  | { type: 'SET_REGENERATING'; payload: boolean }
  | { type: 'SET_STOPPING'; payload: boolean }
  | { type: 'SET_AUTO_SENT'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET_INPUT' }
  | { type: 'START_SEND' }
  | { type: 'FINISH_SEND' }
  | { type: 'START_REGENERATE' }
  | { type: 'FINISH_REGENERATE' }

// ============================================================================
// Reducer
// ============================================================================

/**
 * Reducer function for chat UI state.
 */
function chatUIReducer(state: ChatUIState, action: ChatUIAction): ChatUIState {
  switch (action.type) {
    case 'SET_INPUT':
      return { ...state, input: action.payload }

    case 'SET_COPIED_ID':
      return { ...state, copiedId: action.payload }

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }

    case 'SET_LOADING_THREAD':
      return { ...state, isLoadingThread: action.payload }

    case 'SET_REGENERATING':
      return { ...state, isRegenerating: action.payload }

    case 'SET_STOPPING':
      return { ...state, isStopping: action.payload }

    case 'SET_AUTO_SENT':
      return { ...state, hasAutoSent: action.payload }

    case 'SET_ERROR':
      return { ...state, errorMessage: action.payload }

    case 'RESET_INPUT':
      return { ...state, input: '' }

    case 'START_SEND':
      return {
        ...state,
        input: '',
        isLoading: true,
      }

    case 'FINISH_SEND':
      return {
        ...state,
        isLoading: false,
        isStopping: false,
      }

    case 'START_REGENERATE':
      return {
        ...state,
        isRegenerating: true,
      }

    case 'FINISH_REGENERATE':
      return {
        ...state,
        isRegenerating: false,
        isStopping: false,
      }

    default:
      return state
  }
}

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
