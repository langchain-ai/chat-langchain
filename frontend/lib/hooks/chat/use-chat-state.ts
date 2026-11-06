import { useReducer, useCallback } from "react"
import { MAX_INPUT_CHARS, STORAGE_KEYS } from "../../constants/features"

import { chatUIReducer } from "../../utils/chat/composer-state"
export type { ChatUIState, ChatUIAction } from "../../utils/chat/composer-state"

/** Keep the current thread's draft in localStorage alongside transient chat state. */
export function useChatState(threadId: string) {
  // Initialize input from localStorage if available
  const initialInput =
    typeof window !== "undefined"
      ? (
          localStorage.getItem(`${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`) || ""
        ).slice(0, MAX_INPUT_CHARS)
      : ""

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
  const setInput = useCallback(
    (value: string) => {
      const cappedValue = value.slice(0, MAX_INPUT_CHARS)
      dispatch({ type: "SET_INPUT", payload: cappedValue })

      // Auto-save draft to localStorage
      if (typeof window !== "undefined") {
        if (cappedValue) {
          localStorage.setItem(
            `${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`,
            cappedValue
          )
        } else {
          localStorage.removeItem(`${STORAGE_KEYS.DRAFT_PREFIX}${threadId}`)
        }
      }
    },
    [threadId]
  )

  return {
    state,
    dispatch,
    setInput,
  }
}
