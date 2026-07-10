import { useState, useCallback } from "react"
import type { Message } from "../../types"
import { FEEDBACK_KEY } from "../../constants/features"
import { createOrUpdateFeedback, deleteFeedback } from "../../api/langsmith"

interface UseFeedbackProps {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
}

const DEFAULT_POSITIVE_COMMENT = "User rated this as helpful"
const DEFAULT_NEGATIVE_COMMENT = "User indicated the answer wasn't satisfactory"

/**
 * Custom hook for managing user feedback on LangSmith traces.
 * Handles creation, update, and deletion of feedback with local state management.
 * Uses server-side API routes to keep LangSmith API keys secure.
 */
export function useFeedback({ messages, setMessages }: UseFeedbackProps) {
  const [feedbackComment, setFeedbackComment] = useState<{ [messageId: string]: string }>({})
  const [showCommentInput, setShowCommentInput] = useState<string | null>(null)

  /**
   * Applies feedback state changes to the messages array.
   */
  const applyFeedback = useCallback((
    messageId: string,
    value: "positive" | "negative" | null,
    opts?: { id?: string | null; comment?: string | null }
  ) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId
          ? {
              ...m,
              feedback: value,
              ...(opts?.id !== undefined ? { feedbackId: opts.id ?? undefined } : {}),
              ...(opts?.comment !== undefined ? { feedbackComment: opts.comment ?? undefined } : {}),
            }
          : m
      )
    )
  }, [setMessages])

  /**
   * Main feedback handler - creates, updates, or deletes feedback in LangSmith.
   * Supports toggling feedback and adding comments.
   */
  const handleFeedback = useCallback(async (
    messageId: string,
    feedbackType: "positive" | "negative",
    comment?: string,
    options?: { toggle?: boolean }
  ) => {
    const message = messages.find((m) => m.id === messageId)
    if (!message || !message.runId) {
      console.warn("Cannot submit feedback: missing runId on message")
      return
    }

    const shouldToggle = options?.toggle ?? true
    const isSameFeedback = message.feedback === feedbackType
    const trimmedComment = comment?.trim()
    const hasComment = Boolean(trimmedComment)
    const commentPayload = hasComment
      ? trimmedComment!
      : feedbackType === "positive"
        ? DEFAULT_POSITIVE_COMMENT
        : DEFAULT_NEGATIVE_COMMENT

    // Save previous state for rollback on error
    const previousFeedback = message.feedback ?? null
    const previousFeedbackId = message.feedbackId ?? undefined
    const previousComment = message.feedbackComment ?? null

    // Handle toggle-off (remove feedback)
    if (shouldToggle && isSameFeedback && !hasComment) {
      applyFeedback(messageId, null, { id: null, comment: null })

      setFeedbackComment((prev) => {
        const updated = { ...prev }
        delete updated[messageId]
        return updated
      })

      if (showCommentInput === messageId) {
        setShowCommentInput(null)
      }

      // Delete from LangSmith if it exists
      if (message.feedbackId) {
        try {
          await deleteFeedback(message.feedbackId)
        } catch (error) {
          console.error("Error deleting feedback:", error)
          // Rollback on error
          applyFeedback(messageId, previousFeedback, {
            ...(previousFeedbackId !== undefined ? { id: previousFeedbackId } : {}),
            comment: previousComment,
          })
          setFeedbackComment((prev) => {
            const updated = { ...prev }
            if (previousComment != null) {
              updated[messageId] = previousComment
            } else {
              delete updated[messageId]
            }
            return updated
          })
          return
        }
      }

      return
    }

    // Apply feedback optimistically
    applyFeedback(messageId, feedbackType, hasComment ? { comment: commentPayload } : undefined)

    try {
      let feedbackId = message.feedbackId

      // Create or update feedback via API
      // If update fails with 404 (feedback deleted), create new one
      const result = await createOrUpdateFeedback({
        runId: message.runId,
        feedbackKey: FEEDBACK_KEY,
        score: feedbackType,
        comment: hasComment ? commentPayload : previousComment || undefined,
        feedbackId: feedbackId,
      })
      feedbackId = result.id

      // Update state with feedback ID from server
      if (feedbackId && feedbackId !== previousFeedbackId) {
        applyFeedback(messageId, feedbackType, { id: feedbackId })
      }

      // Update comment state
      if (hasComment) {
        applyFeedback(messageId, feedbackType, { comment: commentPayload })
        setFeedbackComment((prev) => ({
          ...prev,
          [messageId]: commentPayload,
        }))

        if (showCommentInput === messageId) {
          setShowCommentInput(null)
        }
      }

    } catch (error) {
      console.error("Error submitting feedback:", error)

      // Rollback optimistic update on error
      applyFeedback(messageId, previousFeedback, {
        ...(previousFeedbackId !== undefined ? { id: previousFeedbackId } : {}),
        comment: previousComment,
      })

      if (hasComment) {
        setFeedbackComment((prev) => ({
          ...prev,
          [messageId]: commentPayload,
        }))
      }
    }
  }, [messages, showCommentInput, applyFeedback])

  /**
   * Submits a comment for existing feedback.
   */
  const handleSubmitComment = useCallback(async (messageId: string) => {
    const message = messages.find((m) => m.id === messageId)
    const comment = feedbackComment[messageId]?.trim()

    if (!message || !message.feedback || !comment) {
      console.warn("Cannot submit comment: ensure feedback is selected and comment is provided")
      return
    }

    await handleFeedback(messageId, message.feedback, comment, { toggle: false })
  }, [messages, feedbackComment, handleFeedback])

  /**
   * Cancels comment input and clears the draft.
   */
  const handleCancelComment = useCallback((messageId: string) => {
    setShowCommentInput((prev) => (prev === messageId ? null : prev))
    setFeedbackComment((prev) => {
      const updated = { ...prev }
      delete updated[messageId]
      return updated
    })
  }, [])

  /**
   * Toggles the comment input visibility for a message.
   */
  const handleToggleComment = useCallback((messageId: string) => {
    if (showCommentInput !== messageId) {
      const message = messages.find((m) => m.id === messageId)
      setFeedbackComment((prev) => ({
        ...prev,
        [messageId]: prev[messageId] ?? message?.feedbackComment ?? "",
      }))
    }

    setShowCommentInput((prev) => (prev === messageId ? null : messageId))
  }, [messages, showCommentInput])

  return {
    feedbackComment,
    showCommentInput,
    handleFeedback,
    handleSubmitComment,
    handleCancelComment,
    handleToggleComment,
    setFeedbackComment,
  }
}
