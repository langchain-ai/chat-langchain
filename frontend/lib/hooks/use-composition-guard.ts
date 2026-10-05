import { useCallback, useEffect, useRef } from "react"

interface CompositionKeyboardEvent {
  nativeEvent: {
    isComposing?: boolean
  }
  keyCode?: number
}

export function shouldIgnoreCompositionKeyDown(
  event: CompositionKeyboardEvent,
  isComposing: boolean,
): boolean {
  return isComposing || event.nativeEvent.isComposing === true || event.keyCode === 229
}

export function useCompositionGuard() {
  const isComposingRef = useRef(false)
  const compositionEndTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearCompositionEndTimeout = useCallback(() => {
    if (compositionEndTimeoutRef.current !== null) {
      clearTimeout(compositionEndTimeoutRef.current)
      compositionEndTimeoutRef.current = null
    }
  }, [])

  const onCompositionStart = useCallback(() => {
    clearCompositionEndTimeout()
    isComposingRef.current = true
  }, [clearCompositionEndTimeout])

  const onCompositionEnd = useCallback(() => {
    clearCompositionEndTimeout()
    isComposingRef.current = true
    compositionEndTimeoutRef.current = setTimeout(() => {
      isComposingRef.current = false
      compositionEndTimeoutRef.current = null
    }, 0)
  }, [clearCompositionEndTimeout])

  const shouldIgnoreKeyDown = useCallback(
    (event: CompositionKeyboardEvent) =>
      shouldIgnoreCompositionKeyDown(event, isComposingRef.current),
    [],
  )

  useEffect(() => clearCompositionEndTimeout, [clearCompositionEndTimeout])

  return { onCompositionStart, onCompositionEnd, shouldIgnoreKeyDown }
}
