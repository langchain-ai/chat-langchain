import { memo, useMemo, useEffect, useRef, useState, useCallback } from "react"
import type { Message } from "@/lib/types"
import { MessageItem } from "./message-item"
import { ArrowDown } from "lucide-react"

interface MessageListProps {
  messages: Message[]
  showToolCalls?: boolean
  isRegenerating: boolean
  copiedId: string | null
  onCopy: (content: string, messageId: string) => void
  onRegenerate: () => void
  onEditAndRerun?: (messageId: string, newContent: string) => void
  feedbackComment: { [messageId: string]: string }
  showCommentInput: string | null
  onFeedback: (messageId: string, feedbackType: "positive" | "negative", comment?: string) => void
  onSubmitComment: (messageId: string) => void
  onCancelComment: (messageId: string) => void
  onToggleComment: (messageId: string) => void
  setFeedbackComment: React.Dispatch<React.SetStateAction<{ [messageId: string]: string }>>
}

export const MessageList = memo(function MessageList({
  messages,
  showToolCalls,
  isRegenerating,
  copiedId,
  onCopy,
  onRegenerate,
  onEditAndRerun,
  feedbackComment,
  showCommentInput,
  onFeedback,
  onSubmitComment,
  onCancelComment,
  onToggleComment,
  setFeedbackComment,
}: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const shouldAutoScrollRef = useRef(true)
  const lastMessageCountRef = useRef(0)
  const lastContentRef = useRef("")
  const isProgrammaticScrollRef = useRef(false)
  const firstMessageIdRef = useRef<string | null>(null)
  const scrollTimeoutRef = useRef<NodeJS.Timeout>()
  const mutationObserverRef = useRef<MutationObserver | null>(null)
  const scrollAttemptsRef = useRef(0)
  const isAutoScrollingRef = useRef(false)
  const lastScrollTopRef = useRef(0)

  // Memoize lastAssistantId calculation
  const lastAssistantId = useMemo(() => {
    const assistantMessages = messages.filter((m) => m.role === "assistant")
    return assistantMessages.length > 0 ? assistantMessages[assistantMessages.length - 1]?.id : undefined
  }, [messages])

  /**
   * Cancels the auto-scroll process when user manually scrolls up
   * Cleans up all observers and timeouts
   */
  const cancelAutoScroll = useCallback(() => {
    isAutoScrollingRef.current = false
    isProgrammaticScrollRef.current = false

    if (mutationObserverRef.current) {
      mutationObserverRef.current.disconnect()
      mutationObserverRef.current = null
    }
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current)
    }
  }, [])

  /**
   * Scrolls to the absolute bottom of the message list instantly
   * Updates last scroll position to prevent false positive user scroll detection
   */
  const scrollToAbsoluteBottom = useCallback(() => {
    if (!scrollRef.current) return

    const maxScroll = scrollRef.current.scrollHeight
    scrollRef.current.scrollTop = maxScroll
    lastScrollTopRef.current = maxScroll
  }, [])

  /**
   * Checks if scroll position is at the absolute bottom (within 5px tolerance)
   */
  const isAtAbsoluteBottom = useCallback(() => {
    if (!scrollRef.current) return true
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    return scrollTop >= scrollHeight - clientHeight - 5
  }, [])

  /**
   * Auto-scroll to bottom when switching threads or on initial load
   *
   * Features:
   * - Monitors DOM changes with MutationObserver to scroll as content renders
   * - Detects when content height stabilizes to know when rendering is complete
   * - Verifies we're actually at bottom before stopping (handles lazy-loaded content)
   * - Supports very long chats (up to 10 seconds of monitoring)
   * - Can be cancelled if user scrolls up manually
   */
  useEffect(() => {
    if (!scrollRef.current || messages.length === 0) return

    const currentFirstMessageId = messages[0]?.id
    const isInitialLoad = firstMessageIdRef.current === null
    const threadChanged = firstMessageIdRef.current !== null && firstMessageIdRef.current !== currentFirstMessageId

    if (isInitialLoad || threadChanged) {
      isProgrammaticScrollRef.current = true
      isAutoScrollingRef.current = true
      shouldAutoScrollRef.current = true
      scrollAttemptsRef.current = 0

      // Clean up any existing observers/timeouts
      if (mutationObserverRef.current) {
        mutationObserverRef.current.disconnect()
        mutationObserverRef.current = null
      }
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }

      const scrollContainer = scrollRef.current
      let lastScrollHeight = 0
      let stabilityCheckCount = 0
      const MAX_SCROLL_ATTEMPTS = 100 // 10 seconds max (100 × 100ms)
      const STABILITY_THRESHOLD = 5 // 500ms of no changes required
      const CHECK_INTERVAL = 100

      /**
       * Continuously scrolls to bottom and checks if content has finished rendering
       * Stops when: content is stable AND we're at absolute bottom OR user cancels OR max attempts reached
       */
      const scrollAndCheck = () => {
        // Stop if user cancelled by scrolling up
        if (!isAutoScrollingRef.current) return

        if (!scrollContainer || scrollAttemptsRef.current >= MAX_SCROLL_ATTEMPTS) {
          // Max attempts reached - perform final cleanup
          scrollToAbsoluteBottom()
          setTimeout(() => {
            scrollToAbsoluteBottom()
            isProgrammaticScrollRef.current = false
            isAutoScrollingRef.current = false
            mutationObserverRef.current?.disconnect()
          }, 200)
          return
        }

        scrollAttemptsRef.current++
        const currentScrollHeight = scrollContainer.scrollHeight

        // Scroll to current bottom
        scrollToAbsoluteBottom()

        // Check if content height has stabilized
        if (currentScrollHeight === lastScrollHeight) {
          stabilityCheckCount++

          if (stabilityCheckCount >= STABILITY_THRESHOLD) {
            // Verify we're actually at the bottom before stopping
            const currentScrollTop = scrollContainer.scrollTop
            const maxScrollTop = scrollContainer.scrollHeight - scrollContainer.clientHeight
            const distanceFromBottom = maxScrollTop - currentScrollTop
            const isAtBottom = distanceFromBottom <= 10

            if (isAtBottom) {
              // Content stable AND at bottom - safe to stop
              setTimeout(() => {
                scrollToAbsoluteBottom()
                isProgrammaticScrollRef.current = false
                isAutoScrollingRef.current = false
                mutationObserverRef.current?.disconnect()
              }, 200)
              return
            } else {
              // Not at bottom yet - continue monitoring
              stabilityCheckCount = 0
            }
          }
        } else {
          // Content height changed - reset stability counter
          stabilityCheckCount = 0
          lastScrollHeight = currentScrollHeight
        }

        // Schedule next check
        scrollTimeoutRef.current = setTimeout(scrollAndCheck, CHECK_INTERVAL)
      }

      /**
       * Set up MutationObserver to detect DOM changes in real-time
       * Scrolls immediately when new content is added
       */
      mutationObserverRef.current = new MutationObserver((mutations) => {
        if (mutations.length > 0) {
          scrollToAbsoluteBottom()
          stabilityCheckCount = 0 // Reset stability counter
        }
      })

      mutationObserverRef.current.observe(scrollContainer, {
        childList: true,      // New elements added/removed
        subtree: true,        // Watch entire DOM tree
        attributes: true,     // Style/class changes
        characterData: true,  // Text content changes
      })

      // Perform immediate first scroll
      scrollToAbsoluteBottom()

      // Start the monitoring loop after initial render
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          scrollTimeoutRef.current = setTimeout(scrollAndCheck, 150)
        })
      })
    }

    firstMessageIdRef.current = currentFirstMessageId
  }, [messages, scrollToAbsoluteBottom, isAtAbsoluteBottom])

  /**
   * Cleanup observers and timeouts on unmount
   */
  useEffect(() => {
    return () => {
      if (mutationObserverRef.current) {
        mutationObserverRef.current.disconnect()
      }
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }
    }
  }, [])

  /**
   * Checks if scroll position is at bottom (within 1px tolerance)
   */
  const isAtBottom = useCallback(() => {
    if (!scrollRef.current) return true
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight
    return distanceFromBottom < 1
  }, [])

  /**
   * Handles user scroll events
   * - Detects upward scrolling during auto-scroll and cancels it
   * - Updates scroll button visibility
   * - Manages auto-scroll state based on position
   */
  const handleScroll = useCallback(() => {
    if (!scrollRef.current || isProgrammaticScrollRef.current) return

    const currentScrollTop = scrollRef.current.scrollTop
    const atBottom = isAtBottom()

    // Cancel auto-scroll if user scrolls up during auto-scroll
    if (isAutoScrollingRef.current) {
      const scrolledUp = currentScrollTop < lastScrollTopRef.current

      if (scrolledUp) {
        cancelAutoScroll()
        shouldAutoScrollRef.current = false
      }
    }

    lastScrollTopRef.current = currentScrollTop
    setShowScrollButton(!atBottom)
    shouldAutoScrollRef.current = atBottom
  }, [isAtBottom, cancelAutoScroll])

  /**
   * Auto-scroll during message streaming
   * Only scrolls if user is at bottom and hasn't manually scrolled up
   */
  useEffect(() => {
    if (!scrollRef.current) return

    const lastMessage = messages[messages.length - 1]
    const currentContent = lastMessage?.content || ""
    const isNewMessage = messages.length > lastMessageCountRef.current
    const isStreaming = currentContent !== lastContentRef.current && !isNewMessage

    if (shouldAutoScrollRef.current && (isNewMessage || isStreaming)) {
      isProgrammaticScrollRef.current = true

      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: isStreaming ? 'instant' : 'auto'
      })

      requestAnimationFrame(() => {
        isProgrammaticScrollRef.current = false
      })
    }

    lastMessageCountRef.current = messages.length
    lastContentRef.current = currentContent
  }, [messages])

  /**
   * Scrolls to bottom when user clicks the scroll-to-bottom button
   * Uses smooth animation for better UX
   */
  const scrollToBottom = useCallback(() => {
    if (!scrollRef.current) return
    isProgrammaticScrollRef.current = true
    shouldAutoScrollRef.current = true
    setShowScrollButton(false)

    const targetScroll = scrollRef.current.scrollHeight - scrollRef.current.clientHeight
    scrollRef.current.scrollTo({
      top: targetScroll,
      behavior: 'smooth'
    })

    // Final scroll after animation to ensure absolute bottom
    setTimeout(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight
      }
      isProgrammaticScrollRef.current = false
    }, 400)
  }, [])


  return (
    <>
      <style jsx>{`
        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(20px) scale(0.98);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideInButton {
          from {
            opacity: 0;
            transform: translateY(10px) scale(0.9);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        .scroll-button {
          animation: slideInButton 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
      `}</style>
      <div
        className="flex-1 overflow-y-auto custom-scrollbar relative"
        ref={scrollRef}
        onScroll={handleScroll}
        style={{
          willChange: 'scroll-position',
          contain: 'layout style paint',
          WebkitOverflowScrolling: 'touch',
        }}
      >
        <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">
          {messages.map((message, idx) => {
            const isLastMessage = idx === messages.length - 1
            return (
              <div
                key={message.id}
                style={{
                  animation: isLastMessage && message.role === 'user' ? 'slideInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)' : 'none',
                }}
              >
              <MessageItem
                message={message}
                showToolCalls={showToolCalls}
                isLastAssistant={message.id === lastAssistantId}
                isRegenerating={isRegenerating}
                copiedId={copiedId}
                onCopy={onCopy}
                onRegenerate={onRegenerate}
                onEditAndRerun={onEditAndRerun}
                feedbackComment={feedbackComment}
                showCommentInput={showCommentInput}
                onFeedback={onFeedback}
                onSubmitComment={onSubmitComment}
                onCancelComment={onCancelComment}
                onToggleComment={onToggleComment}
                setFeedbackComment={setFeedbackComment}
              />
            </div>
            )
          })}
        </div>
      </div>

      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="scroll-button fixed bottom-32 right-4 sm:right-8 p-3 rounded-full shadow-lg hover:scale-110 active:scale-95 transition-transform z-50"
          style={{
            background: '#7FC8FF',
            color: 'white',
          }}
          aria-label="Scroll to bottom"
        >
          <ArrowDown className="w-5 h-5" />
        </button>
      )}
    </>
  )
})
