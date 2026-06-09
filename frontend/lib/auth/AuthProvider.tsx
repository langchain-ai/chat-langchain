"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react"
import type { AuthChangeEvent, Session, User } from "@supabase/supabase-js"
import {
  type AuthRegion,
  getAvailableAuthRegions,
  getStoredAuthRegion,
  getSupabaseClient,
  isSupabaseAuthConfigured,
  signOutAllSupabaseClients,
  setStoredAuthRegion,
} from "./supabase"

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

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authRegion, setAuthRegionState] = useState<AuthRegion>(() =>
    getStoredAuthRegion()
  )
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const availableAuthRegions = getAvailableAuthRegions()
  const isConfigured = isSupabaseAuthConfigured(authRegion)

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
      (event: AuthChangeEvent, session: Session | null) => {
        if (
          event === "SIGNED_IN" ||
          event === "TOKEN_REFRESHED" ||
          event === "INITIAL_SESSION"
        ) {
          setUser(session?.user ?? null)
          setSession(session ?? null)
        }

        if (event === "SIGNED_OUT") {
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
