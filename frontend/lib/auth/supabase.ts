"use client"

import { createBrowserClient } from "@supabase/ssr"
import type { SupabaseClient } from "@supabase/supabase-js"

export type AuthRegion = "us" | "eu" | "apac" | "aws"

export const AUTH_REGION_LABELS: Record<AuthRegion, string> = {
  us: "US",
  eu: "EU",
  apac: "APAC",
  aws: "AWS US",
}

const AUTH_REGION_STORAGE_KEY = "chat-langchain-auth-region"

const supabaseConfigs: Record<AuthRegion, { url?: string; anonKey?: string }> = {
  us: {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  },
  eu: {
    url: process.env.NEXT_PUBLIC_SUPABASE_EU_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_EU_ANON_KEY,
  },
  apac: {
    url: process.env.NEXT_PUBLIC_SUPABASE_APAC_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_APAC_ANON_KEY,
  },
  aws: {
    url: process.env.NEXT_PUBLIC_SUPABASE_AWS_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_AWS_ANON_KEY,
  },
}

const clients = new Map<AuthRegion, SupabaseClient>()

export const isAuthRegion = (value: string | null | undefined): value is AuthRegion =>
  value === "us" || value === "eu" || value === "apac" || value === "aws"

export const isSupabaseAuthConfigured = (region: AuthRegion = "us"): boolean => {
  const config = supabaseConfigs[region]
  return Boolean(config.url && config.anonKey)
}

export const getAvailableAuthRegions = (): AuthRegion[] =>
  (Object.keys(supabaseConfigs) as AuthRegion[]).filter(isSupabaseAuthConfigured)

export const getDefaultAuthRegion = (): AuthRegion => {
  const availableRegions = getAvailableAuthRegions()
  return availableRegions[0] ?? "us"
}

export const getStoredAuthRegion = (): AuthRegion => {
  if (typeof window === "undefined") return getDefaultAuthRegion()

  const storedRegion = window.localStorage.getItem(AUTH_REGION_STORAGE_KEY)
  if (isAuthRegion(storedRegion)) {
    return storedRegion
  }

  return getDefaultAuthRegion()
}

export const setStoredAuthRegion = (region: AuthRegion): void => {
  if (typeof window === "undefined") return
  window.localStorage.setItem(AUTH_REGION_STORAGE_KEY, region)
}

export const getSupabaseClient = (
  region: AuthRegion = getStoredAuthRegion()
): SupabaseClient | null => {
  const existingClient = clients.get(region)
  if (existingClient) return existingClient

  const config = supabaseConfigs[region]
  if (!config.url || !config.anonKey) {
    console.warn(`[Auth] Supabase auth credentials are not configured for ${region}`)
    return null
  }

  const client = createBrowserClient(config.url, config.anonKey)
  clients.set(region, client)
  return client
}
