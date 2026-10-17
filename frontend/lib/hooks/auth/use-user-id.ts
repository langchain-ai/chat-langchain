/**
 * User ID Management Hook
 *
 * Public Chat LangChain uses an authenticated email when available and falls
 * back to a verified Supabase anonymous user.
 */

'use client'

import { useState, useEffect } from 'react'
import { useAuth, type AuthRegion } from '@/lib/auth'

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

export function useGuestAuth(authRegion: AuthRegion): GuestAuthState {
  const [guestAuth, setGuestAuth] = useState<GuestAuthState>({
    guestUserId: null,
    guestToken: null,
    loading: true,
  })

  useEffect(() => {
    let cancelled = false

    const loadGuestAuth = async () => {
      setGuestAuth({
        guestUserId: null,
        guestToken: null,
        loading: true,
      })
      try {
        const response = await fetch('/api/auth/guest', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ region: authRegion }),
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

        console.warn('[Auth] Guest authentication unavailable:', error)
        if (!cancelled) {
          setGuestAuth({
            guestUserId: null,
            guestToken: null,
            loading: false,
          })
        }
      }
    }

    void loadGuestAuth()

    return () => {
      cancelled = true
    }
  }, [authRegion])

  return guestAuth
}

export function useLangGraphAuth(): LangGraphAuthState {
  const { user, session, loading: authLoading, authRegion } = useAuth()
  const { guestUserId, guestToken, loading: guestLoading } =
    useGuestAuth(authRegion)
  const signedInUserId = user?.email ?? null
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
