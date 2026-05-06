/**
 * String Utilities
 *
 * String manipulation and formatting functions.
 */

/**
 * Truncate text to a maximum length.
 * Appends "..." if truncated.
 */
export const truncate = (text: string, max: number): string =>
  text.length > max ? `${text.slice(0, max)}...` : text

/**
 * Thread Title Generation
 *
 * Generates concise, descriptive titles for conversation threads by calling
 * the backend API endpoint which uses Google Gemini for AI-powered titles.
 *
 * - Backend title generation (secure, server-side API key)
 * - Heuristic-based quick titles (instant, no API call)
 * - Fallback truncation (when API is unavailable)
 */

// ============================================================================
// Constants
// ============================================================================

import { LANGGRAPH_API_URL } from "../../constants/api"
import { DEFAULT_TITLE_MAX_LENGTH } from "../../constants/features"

const DEFAULT_MAX_LENGTH = DEFAULT_TITLE_MAX_LENGTH

// ============================================================================
// Types
// ============================================================================

interface TitleGenerationOptions {
  userMessage: string
  assistantResponse?: string
  maxLength?: number
}

interface TitleGenerationResponse {
  title: string
}

// ============================================================================
// Backend API Title Generation
// ============================================================================

/**
 * Generate a smart title for a conversation thread using backend API.
 * Falls back to truncated user message if API fails or is unavailable.
 *
 * The backend uses Google Gemini for AI-powered title generation,
 * keeping API keys secure on the server side.
 *
 * @param options - Configuration for title generation
 * @returns A concise, descriptive title (max 60 chars by default)
 */
export async function generateThreadTitle({
  userMessage,
  assistantResponse,
  maxLength = DEFAULT_MAX_LENGTH,
}: TitleGenerationOptions): Promise<string> {
  // Fallback: use truncated message if no API URL configured
  if (!LANGGRAPH_API_URL) {
    console.warn("No LangGraph API URL configured, using fallback title")
    return truncateTitle(userMessage, maxLength)
  }

  try {
    const response = await fetch(`${LANGGRAPH_API_URL}/generate-title`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userMessage,
        assistantResponse,
        maxLength,
      }),
    })

    if (!response.ok) {
      console.error("Title generation API error:", response.status)
      return truncateTitle(userMessage, maxLength)
    }

    const data: TitleGenerationResponse = await response.json()
    const generatedTitle = data.title?.trim()

    if (!generatedTitle) {
      return truncateTitle(userMessage, maxLength)
    }

    return generatedTitle
  } catch (error) {
    console.error("Error generating title:", error)
    return truncateTitle(userMessage, maxLength)
  }
}

// ============================================================================
// Heuristic-Based Quick Title Generation
// ============================================================================

/**
 * Generate a quick title using pattern matching (no API call).
 * Use this for instant titles, then optionally upgrade with AI in background.
 *
 * @param userMessage - The user's message
 * @returns A heuristic-based title
 */
export function generateQuickTitle(userMessage: string): string {
  const message = userMessage.trim().toLowerCase()

  // Pattern: "How to..." questions
  if (message.match(/^(how do i|how to|how can i)/i)) {
    const extracted = message.replace(/^(how do i|how to|how can i)\s+/i, "")
    return `How to ${extracted.slice(0, 45)}`
  }

  // Pattern: "What is..." questions
  if (message.match(/^(what is|what are|what's)/i)) {
    const extracted = message.replace(/^(what is|what are|what's)\s+/i, "")
    return `About ${extracted.slice(0, 50)}`
  }

  // Pattern: Other question words
  if (message.match(/^(why|when|where)/i)) {
    return truncateTitle(message, 60)
  }

  // Pattern: Error/issue mentions
  if (message.match(/error|issue|problem|bug/i)) {
    const errorMatch = message.match(/(error|issue|problem|bug)[:\s]+([^.!?]+)/i)
    if (errorMatch) {
      return `Error: ${errorMatch[2].slice(0, 50)}`
    }
    return "Error Investigation"
  }

  // Default: truncate
  return truncateTitle(userMessage, 60)
}

// ============================================================================
// Text Cleaning Utilities
// ============================================================================

/**
 * Fallback: Truncate user message to use as title.
 * - Removes common question prefixes
 * - Capitalizes first letter
 * - Truncates to max length
 */
function truncateTitle(message: string, maxLength: number): string {
  let title = message.trim()

  // Remove common prefixes
  title = title.replace(
    /^(how do i|how to|can you|please|help me with|i need help with)\s+/i,
    ""
  )

  // Capitalize first letter
  if (title) {
    title = title.charAt(0).toUpperCase() + title.slice(1)
  }

  // Truncate if needed
  if (title.length > maxLength) {
    title = title.slice(0, maxLength - 3) + "..."
  }

  return title
}
