/**
 * Message Creation and Manipulation Utilities
 *
 * Functions for creating and updating messages in chat conversations.
 */

import type { Message } from "../../types"

// ============================================================================
// Message ID Generation
// ============================================================================

let messageIdCounter = 0

/**
 * Generate a unique message ID.
 * Uses timestamp + counter to ensure uniqueness even for rapid message creation.
 */
export const generateMessageId = (): string => {
  const timestamp = Date.now()
  const counter = messageIdCounter++
  return `${timestamp}-${counter}`
}

// ============================================================================
// Message Creation
// ============================================================================

/**
 * Create a new user message.
 */
export const createUserMessage = (content: string): Message => ({
  id: generateMessageId(),
  role: "user",
  content,
  timestamp: new Date(),
})

// ============================================================================
// Message List Manipulation
// ============================================================================

/**
 * Update a specific message in a message list.
 * Returns a new array with the updated message.
 */
export const updateMessageInList = (
  messages: Message[],
  messageId: string,
  updates: Partial<Message>
): Message[] => {
  return messages.map((m) => (m.id === messageId ? { ...m, ...updates } : m))
}

/**
 * Ensure a message exists in the list.
 * If the message doesn't exist, appends it to the end.
 */
export const ensureMessageExists = (
  messages: Message[],
  messageId: string,
  baseMessage: Message
): Message[] => {
  const existing = messages.find((m) => m.id === messageId)
  return existing ? messages : [...messages, baseMessage]
}

