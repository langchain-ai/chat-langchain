"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react"
import type { AuthChangeEvent, Session, User } from "@supabase/supabase-js"
import {
  type AuthRegion,
  getAvailableAuthRegions,
  getStoredAuthRegion,
  getSupabaseClient,
  isAuthRegion,
  isSupabaseAuthConfigured,
  signOutAllSupabaseClients,
  setStoredAuthRegion,
} from "./supabase"
import {
  claimSignInTracking,
  getSignInFingerprint,
  rememberSignInTracking,
} from "./sign-in-tracking"

export type OAuthProvider = "google" | "github" | "discord"

interface AuthContextType {
  user: User | null
  session: Session | null
  loading: boolean
  isConfigured: boolean
  authRegion: AuthRegion
  availableAuthRegions: AuthRegion[]
  setAuthRegion: (region: AuthRegion) => void
  signIn: (provider: OAuthProvider) => Promise<void>
  signInWithEmail: (email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)
const OAUTH_SUCCESS_PARAM = "auth_success"
const OAUTH_REGION_PARAM = "auth_region"

function getOAuthSignInRegion(): AuthRegion | null {
  if (typeof window === "undefined") return null
  const params = new URL(window.location.href).searchParams
  if (params.get(OAUTH_SUCCESS_PARAM) !== "1") return null
  const region = params.get(OAUTH_REGION_PARAM)
  return isAuthRegion(region) && isSupabaseAuthConfigured(region) ? region : null
}

function hasOAuthSignInMarker(region: AuthRegion): boolean {
  return getOAuthSignInRegion() === region
}

function consumeOAuthSignInMarker(region: AuthRegion): boolean {
  if (!hasOAuthSignInMarker(region)) return false
  const url = new URL(window.location.href)
  url.searchParams.delete(OAUTH_SUCCESS_PARAM)
  url.searchParams.delete(OAUTH_REGION_PARAM)
  window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`)
  return true
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authRegion, setAuthRegionState] = useState<AuthRegion>(() =>
    getOAuthSignInRegion() ?? getStoredAuthRegion()
  )
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const availableAuthRegions = getAvailableAuthRegions()
  const isConfigured = isSupabaseAuthConfigured(authRegion)
  const lastTrackedSignInRef = useRef<string | null>(null)

  const setAuthRegion = useCallback((region: AuthRegion) => {
    if (!isSupabaseAuthConfigured(region)) return
    setStoredAuthRegion(region)
    setAuthRegionState(region)
    setUser(null)
    setSession(null)
    setLoading(true)
  }, [])

  useEffect(() => {
    let cancelled = false
    const checkSession = async () => {
      if (!isConfigured) {
        if (!cancelled) {
          setUser(null)
          setSession(null)
          setLoading(false)
        }
        return
      }

      const client = getSupabaseClient(authRegion)
      if (!client) {
        if (!cancelled) {
          setUser(null)
          setSession(null)
          setLoading(false)
        }
        return
      }

      setLoading(true)
      try {
        const {
          data: { session },
        } = await client.auth.getSession()
        if (!cancelled) {
          setSession(session ?? null)
          setUser(session?.user ?? null)
          if (session?.user && !hasOAuthSignInMarker(authRegion)) {
            lastTrackedSignInRef.current = getSignInFingerprint(
              authRegion,
              session.user
            )
            void rememberSignInTracking(authRegion, session.user)
          }
        }
      } catch {
        // Session check failed, user remains null
        if (!cancelled) {
          setUser(null)
          setSession(null)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void checkSession()
    return () => {
      cancelled = true
    }
  }, [authRegion, isConfigured])

  useEffect(() => {
    if (!isConfigured) return

    const client = getSupabaseClient(authRegion)
    if (!client) return

    const {
      data: { subscription },
    } = client.auth.onAuthStateChange(
      async (event: AuthChangeEvent, session: Session | null) => {
        if (
          event === "SIGNED_IN" ||
          event === "TOKEN_REFRESHED" ||
          event === "INITIAL_SESSION"
        ) {
          setUser(session?.user ?? null)
          setSession(session ?? null)

          if (session?.user) {
            const fingerprint = getSignInFingerprint(authRegion, session.user)
            const markedOAuthSignIn = (
              event === "SIGNED_IN" || event === "INITIAL_SESSION"
            ) && consumeOAuthSignInMarker(authRegion)
            const shouldTrack = event === "SIGNED_IN" || markedOAuthSignIn

            if (shouldTrack && lastTrackedSignInRef.current !== fingerprint) {
              lastTrackedSignInRef.current = fingerprint
              if (await claimSignInTracking(authRegion, session.user)) {
                const provider = session.user.app_metadata?.provider ?? "email"
                window.analytics?.track("Signed In", {
                  provider,
                  email: session.user.email,
                  deployment: "public",
                })
              }
            } else if (event === "INITIAL_SESSION" && !markedOAuthSignIn) {
              lastTrackedSignInRef.current = fingerprint
              void rememberSignInTracking(authRegion, session.user)
            }
          }
        }

        if (event === "SIGNED_OUT") {
          lastTrackedSignInRef.current = null
          setUser(null)
          setSession(null)
        }
      }
    )

    return () => subscription.unsubscribe()
  }, [authRegion, isConfigured])

  const requireClient = useCallback(() => {
    const client = getSupabaseClient(authRegion)
    if (!client) {
      throw new Error(`Supabase auth is not configured for ${authRegion}`)
    }
    return client
  }, [authRegion])

  const signIn = useCallback(
    async (provider: OAuthProvider) => {
      const client = requireClient()
      setStoredAuthRegion(authRegion)
      const { error } = await client.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/auth/callback?region=${authRegion}`,
          ...(provider === "google" && {
            queryParams: { prompt: "select_account" },
          }),
        },
      })

      if (error) throw new Error(`Sign in failed: ${error.message}`)
    },
    [authRegion, requireClient]
  )

  const signInWithEmail = useCallback(
    async (email: string, password: string) => {
      const client = requireClient()
      setStoredAuthRegion(authRegion)
      const { error } = await client.auth.signInWithPassword({
        email,
        password,
      })

      if (error) throw new Error(`Sign in failed: ${error.message}`)
    },
    [authRegion, requireClient]
  )

  const signOut = useCallback(async () => {
    await signOutAllSupabaseClients()
    setUser(null)
    setSession(null)
  }, [])

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        loading,
        isConfigured,
        authRegion,
        availableAuthRegions,
        setAuthRegion,
        signIn,
        signInWithEmail,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error("useAuth must be used within AuthProvider")
  return context
}
