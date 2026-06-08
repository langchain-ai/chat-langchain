"use client"

import { createBrowserClient } from "@supabase/ssr"
import type { SupabaseClient } from "@supabase/supabase-js"

const supabaseUsUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseUsAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

let client: SupabaseClient | null = null

export const isSupabaseAuthConfigured = (): boolean =>
  Boolean(supabaseUsUrl && supabaseUsAnonKey)

export const getSupabaseClient = (): SupabaseClient | null => {
  if (client) return client

  if (!supabaseUsUrl || !supabaseUsAnonKey) {
    console.warn("[Auth] Supabase auth credentials are not configured")
    return null
  }

  client = createBrowserClient(supabaseUsUrl, supabaseUsAnonKey)
  return client
}
