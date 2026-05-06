/**
 * Client Profile Hook
 *
 * This hook manages the client profile (user identity) with localStorage persistence.
 * The client profile contains user preferences, settings, and identity information that
 * persists across browser sessions.
 *
 * Key Features:
 * - Automatic localStorage persistence
 * - Initial profile creation with unique identifiers
 * - Profile validation and resolution
 * - Update and reset capabilities
 * - SSR-safe implementation
 */

import { useState, useEffect, useCallback, useRef } from "react"
import type { ClientProfile } from "../threads"
import {
  createClientProfile,
  resolveClientProfile,
} from "@/lib/config/client-config"
import { STORAGE_KEYS } from "../../constants/features"

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = STORAGE_KEYS.CLIENT_PROFILE

// ============================================================================
// Types
// ============================================================================

/**
 * Return type for the useClientProfile hook.
 */
interface UseClientProfileReturn {
  clientProfile: ClientProfile
  hasLoaded: boolean
  updateClientProfile: (updates: Partial<ClientProfile>) => void
  resetClientProfile: () => void
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook to manage client profile with localStorage persistence.
 *
 * On mount, attempts to load the profile from localStorage. If not found,
 * creates a new profile with a unique identifier and persists it.
 * Updates to the profile are automatically saved to localStorage.
 *
 * @returns Object containing the client profile, loading state, and update functions
 *
 * @example
 * ```tsx
 * const { clientProfile, hasLoaded, updateClientProfile, resetClientProfile } = useClientProfile()
 *
 * // Update profile
 * updateClientProfile({ name: "John Doe" })
 *
 * // Reset to new profile
 * resetClientProfile()
 * ```
 */
export function useClientProfile(): UseClientProfileReturn {
  const initialRef = useRef<ClientProfile | null>(null)
  const [clientProfile, setClientProfile] = useState<ClientProfile>(() => {
    const created = createClientProfile()
    initialRef.current = created
    return created
  })
  const [hasLoaded, setHasLoaded] = useState(false)

  // Load profile from localStorage on mount
  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }

    try {
      const stored = window.localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored)
        setClientProfile(resolveClientProfile(parsed))
        setHasLoaded(true)
        return
      }
    } catch (error) {
      console.error("Failed to load client identity", error)
    }

    // No stored profile found, persist the initial one
    if (initialRef.current) {
      try {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify(initialRef.current)
        )
      } catch (error) {
        console.error("Failed to persist client identity", error)
      }
    }

    setHasLoaded(true)
  }, [])

  /**
   * Persists profile to localStorage.
   * Handles errors gracefully and skips during SSR.
   */
  const persist = useCallback((profile: ClientProfile) => {
    if (typeof window === "undefined") {
      return
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(profile))
    } catch (error) {
      console.error("Failed to persist client identity", error)
    }
  }, [])

  /**
   * Updates the client profile with partial updates.
   * Merges updates with existing profile and persists to localStorage.
   */
  const updateClientProfile = useCallback(
    (updates: Partial<ClientProfile>) => {
      setClientProfile((prev) => {
        const merged = resolveClientProfile({ ...prev, ...updates })
        persist(merged)
        return merged
      })
    },
    [persist]
  )

  /**
   * Resets the client profile to a fresh state.
   * Creates a new profile with new unique identifier and persists it.
   */
  const resetClientProfile = useCallback(() => {
    const fresh = createClientProfile()
    setClientProfile(fresh)
    persist(fresh)
  }, [persist])

  return {
    clientProfile,
    hasLoaded,
    updateClientProfile,
    resetClientProfile,
  }
}
