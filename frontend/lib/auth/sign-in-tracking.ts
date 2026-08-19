import type { User } from "@supabase/supabase-js"
import type { AuthRegion } from "./supabase"

type SignInUser = Pick<User, "id" | "last_sign_in_at">
type SignInTrackingStorage = Pick<Storage, "getItem" | "setItem">

interface SignInTrackingLocks {
  request(name: string, callback: () => boolean | Promise<boolean>): Promise<boolean>
}

const STORAGE_KEY_PREFIX = "chat-langchain:last-tracked-sign-in"
const UNKNOWN_SIGN_IN_TIME = "unknown"

export function getSignInFingerprint(
  region: AuthRegion,
  user: SignInUser
): string {
  return `${region}:${user.id}:${user.last_sign_in_at ?? UNKNOWN_SIGN_IN_TIME}`
}

function claimStoredSignIn(
  storage: SignInTrackingStorage,
  storageKey: string
): boolean {
  try {
    if (storage.getItem(storageKey) === "1") return false
    storage.setItem(storageKey, "1")
    return true
  } catch (error) {
    console.warn("[Analytics] Sign-in deduplication storage unavailable", error)
    return true
  }
}

export async function claimSignInTracking(
  region: AuthRegion,
  user: SignInUser,
  storage?: SignInTrackingStorage,
  locks?: SignInTrackingLocks
): Promise<boolean> {
  const fingerprint = getSignInFingerprint(region, user)
  const storageKey = `${STORAGE_KEY_PREFIX}:${encodeURIComponent(fingerprint)}`

  let trackingStorage: SignInTrackingStorage
  try {
    trackingStorage = storage ?? window.localStorage
  } catch (error) {
    console.warn("[Analytics] Sign-in deduplication storage unavailable", error)
    return true
  }

  const claim = () => claimStoredSignIn(trackingStorage, storageKey)
  const lockManager = locks ?? (
    typeof navigator === "undefined" || !navigator.locks
      ? undefined
      : {
          request: (name, callback) => navigator.locks.request(name, callback),
        }
  )
  if (!lockManager) return claim()

  try {
    return await lockManager.request(storageKey, claim)
  } catch (error) {
    console.warn("[Analytics] Sign-in deduplication lock unavailable", error)
    return claim()
  }
}

export async function rememberSignInTracking(
  region: AuthRegion,
  user: SignInUser,
  storage?: SignInTrackingStorage,
  locks?: SignInTrackingLocks
): Promise<void> {
  await claimSignInTracking(region, user, storage, locks)
}
