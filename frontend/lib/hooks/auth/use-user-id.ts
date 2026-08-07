/**
 * User ID Management Hook
 *
 * Public Chat LangChain uses the pseudonymous Supabase uid when available and
 * falls back to an anonymous browser UUID persisted in localStorage. The id is
 * sent as LangSmith trace/thread metadata, so it must never be an email address.
 */

'use client'

import { useState, useEffect } from 'react'
import { useAuth, type AuthRegion } from '@/lib/auth'

// ============================================================================
// Constants
// ============================================================================

const USER_ID_KEY = 'langgraph-user-id'

interface GuestAuthState {
  guestUserId: string | null
  guestToken: string | null
  loading: boolean
}

interface LangGraphAuthState extends GuestAuthState {
  userId: string | null
  authToken: string | null
  authRegion: AuthRegion
}

class GuestAuthRateLimitError extends Error {
  constructor() {
    super('Guest authentication was rate limited')
    this.name = 'GuestAuthRateLimitError'
  }
}

export function getOrCreateGuestUserId(): string | null {
  if (typeof window === 'undefined') return null

  let id = localStorage.getItem(USER_ID_KEY)

  if (!id) {
    id = `user-${crypto.randomUUID()}`
    localStorage.setItem(USER_ID_KEY, id)
    console.info('Generated new browser user ID:', id)
  } else {
    console.info('Loaded existing browser user ID:', id)
  }

  return id
}

/** Opaque id for a signed-in user; never their email or display name. */
export function getPseudonymousUserId(
  user: { id?: string | null } | null | undefined
): string | null {
  return user?.id ? `user-${user.id}` : null
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Get user ID for thread management.
 *
 * Returns the authenticated user's pseudonymous Supabase id or a browser UUID
 * used to filter threads on the LangGraph backend.
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
  const { user, loading } = useAuth()

  useEffect(() => {
    // Skip during SSR
    if (typeof window === 'undefined') return
    if (loading) return

    const signedInUserId = getPseudonymousUserId(user)
    if (signedInUserId) {
      setUserId(signedInUserId)
      return
    }

    setUserId(getOrCreateGuestUserId())
  }, [loading, user])

  return userId
}

export function useGuestUserId(): string | null {
  const [guestUserId, setGuestUserId] = useState<string | null>(null)

  useEffect(() => {
    setGuestUserId(getOrCreateGuestUserId())
  }, [])

  return guestUserId
}

export function useGuestAuth(): GuestAuthState {
  const [guestAuth, setGuestAuth] = useState<GuestAuthState>({
    guestUserId: null,
    guestToken: null,
    loading: true,
  })

  useEffect(() => {
    let cancelled = false

    const loadGuestAuth = async () => {
      try {
        const response = await fetch('/api/auth/guest', {
          method: 'POST',
          credentials: 'same-origin',
        })
        if (response.status === 429) {
          throw new GuestAuthRateLimitError()
        }
        if (!response.ok) {
          throw new Error('Failed to create guest session')
        }

        const data = (await response.json()) as {
          guestId?: string
          token?: string
        }
        if (!data.guestId || !data.token) {
          throw new Error('Guest session response was incomplete')
        }

        if (!cancelled) {
          setGuestAuth({
            guestUserId: data.guestId,
            guestToken: data.token,
            loading: false,
          })
        }
      } catch (error) {
        if (error instanceof GuestAuthRateLimitError) {
          console.warn('[Auth] Guest authentication rate limited; not switching identity')
          if (!cancelled) {
            setGuestAuth({
              guestUserId: null,
              guestToken: null,
              loading: false,
            })
          }
          return
        }

        console.warn(
          '[Auth] Falling back to legacy guest user ID auth:',
          error
        )
        const legacyGuestUserId = getOrCreateGuestUserId()
        if (!cancelled) {
          setGuestAuth({
            guestUserId: legacyGuestUserId,
            guestToken: legacyGuestUserId,
            loading: false,
          })
        }
      }
    }

    void loadGuestAuth()

    return () => {
      cancelled = true
    }
  }, [])

  return guestAuth
}

export function useLangGraphAuth(): LangGraphAuthState {
  const { user, session, loading: authLoading, authRegion } = useAuth()
  const { guestUserId, guestToken, loading: guestLoading } = useGuestAuth()
  const signedInUserId = getPseudonymousUserId(user)
  const signedInToken = session?.access_token ?? null

  return {
    userId: signedInUserId || guestUserId,
    authToken: signedInToken || guestToken,
    authRegion,
    guestUserId,
    guestToken,
    loading: authLoading || (!signedInUserId && guestLoading),
  }
}
