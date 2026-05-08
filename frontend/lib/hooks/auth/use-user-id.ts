/**
 * User ID Management Hook
 *
 * Public Chat LangChain uses an anonymous browser UUID persisted in localStorage.
 */

'use client'

import { useState, useEffect } from 'react'

// ============================================================================
// Constants
// ============================================================================

const USER_ID_KEY = 'langgraph-user-id'

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Get user ID for thread management.
 *
 * Returns a browser UUID used to filter threads on the LangGraph backend.
 *
 * @returns The user's unique ID, or null while loading
 *
 * @example
 * const userId = useUserId()
 * if (!userId) return <div>Loading...</div>
 * return <ChatInterface userId={userId} />
 */
export function useUserId(): string | null {
  const [userId, setUserId] = useState<string | null>(null)

  useEffect(() => {
    // Skip during SSR
    if (typeof window === 'undefined') return

    let id = localStorage.getItem(USER_ID_KEY)

    if (!id) {
      // Generate new ID using crypto.randomUUID()
      id = `user-${crypto.randomUUID()}`
      localStorage.setItem(USER_ID_KEY, id)
      console.info('Generated new browser user ID:', id)
    } else {
      console.info('Loaded existing browser user ID:', id)
    }

    setUserId(id)
  }, [])

  return userId
}
