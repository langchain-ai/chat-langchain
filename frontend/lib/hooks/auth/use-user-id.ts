/**
 * User ID Management Hook
 *
 * Custom React hook for managing user identification.
 *
 * **Internal Deployment (OAuth enabled):**
 * - Uses authenticated user's email from NextAuth session
 * - Persists across devices when signed in
 * - Threads tied to user account, not browser
 *
 * **External Deployment (No auth):**
 * - Falls back to browser-specific UUID
 * - Persists ID in localStorage across sessions
 * - Browser-specific thread management
 */

'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { getDeploymentEnv } from "../../config/deployment-config"

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
 * Returns authenticated user's email (internal) or browser UUID (external).
 * The ID is used to filter threads on the LangGraph backend.
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
  const { data: session, status } = useSession()

  useEffect(() => {
    // Skip during SSR
    if (typeof window === 'undefined') return

    // Check if auth is required from environment variable
    const deploymentEnv = getDeploymentEnv()
    const requiresAuth = deploymentEnv !== 'external'

    // If auth is required (internal deployment), use OAuth user email
    if (requiresAuth) {
      if (status === 'loading') {
        // Still loading session, don't set userId yet
        return
      }

      if (status === 'authenticated' && session?.user?.email) {
        // Use authenticated user's email as the user ID
        const id = session.user.email
        setUserId(id)
        console.info('OAuth authenticated - userId:', id, 'name:', session.user.name)
        return
      }

      // If not authenticated but auth is required, userId remains null
      // (middleware should redirect to sign-in)
      console.warn('Auth required but user not authenticated - userId will be null')
      setUserId(null)
      return
    }

    // External deployment (no auth): Fall back to browser-specific UUID
    let id = localStorage.getItem(USER_ID_KEY)

    if (!id) {
      // Generate new ID using crypto.randomUUID()
      id = `user-${crypto.randomUUID()}`
      localStorage.setItem(USER_ID_KEY, id)
      console.info('Generated new browser user ID (external deployment):', id)
    } else {
      console.info('Loaded existing browser user ID (external deployment):', id)
    }

    setUserId(id)
  }, [status, session?.user?.email])

  return userId
}
