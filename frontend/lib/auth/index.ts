export { AuthProvider, useAuth } from "./AuthProvider"
export type { OAuthProvider } from "./AuthProvider"
export {
  AUTH_REGION_LABELS,
  getAvailableAuthRegions,
  getDefaultAuthRegion,
  getStoredAuthRegion,
  getSupabaseClient,
  isAuthRegion,
  isSupabaseAuthConfigured,
  setStoredAuthRegion,
} from "./supabase"
export type { AuthRegion } from "./supabase"
