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
  getSupabaseClient,
  isSupabaseAuthConfigured,
} from "./supabase"

export type OAuthProvider = "google" | "github" | "discord"

interface AuthContextType {
  user: User | null
  session: Session | null
  loading: boolean
  isConfigured: boolean
  signIn: (provider: OAuthProvider) => Promise<void>
  signInWithEmail: (email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const isConfigured = isSupabaseAuthConfigured()

  useEffect(() => {
    const checkSession = async () => {
      if (!isConfigured) {
        setUser(null)
        setSession(null)
        setLoading(false)
        return
      }

      const client = getSupabaseClient()
      if (!client) {
        setLoading(false)
        return
      }

      try {
        const {
          data: { session },
        } = await client.auth.getSession()
        setSession(session ?? null)
        setUser(session?.user ?? null)
      } catch {
        // Session check failed, user remains null
      } finally {
        setLoading(false)
      }
    }

    void checkSession()
  }, [isConfigured])

  useEffect(() => {
    if (!isConfigured) return

    const client = getSupabaseClient()
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
  }, [isConfigured])

  const requireClient = useCallback(() => {
    const client = getSupabaseClient()
    if (!client) {
      throw new Error("Supabase auth is not configured")
    }
    return client
  }, [])

  const signIn = useCallback(
    async (provider: OAuthProvider) => {
      const client = requireClient()
      const { error } = await client.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
          ...(provider === "google" && {
            queryParams: { prompt: "select_account" },
          }),
        },
      })

      if (error) throw new Error(`Sign in failed: ${error.message}`)
    },
    [requireClient]
  )

  const signInWithEmail = useCallback(
    async (email: string, password: string) => {
      const client = requireClient()
      const { error } = await client.auth.signInWithPassword({
        email,
        password,
      })

      if (error) throw new Error(`Sign in failed: ${error.message}`)
    },
    [requireClient]
  )

  const signOut = useCallback(async () => {
    const client = getSupabaseClient()
    if (!client) {
      setUser(null)
      setSession(null)
      return
    }

    const { error } = await client.auth.signOut()
    if (error && error.message !== "Auth session missing!") {
      throw new Error(`Sign out failed: ${error.message}`)
    }
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
