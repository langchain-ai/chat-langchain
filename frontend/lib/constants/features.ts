/**
 * Feature Constants
 *
 * Application-wide feature flags and limits.
 */

export const THREAD_FETCH_LIMIT = 100

export const DEFAULT_TITLE_MAX_LENGTH = 60

export const MAX_INPUT_CHARS = 50_000

export const INPUT_TOO_LONG_MESSAGE = "input is too long, try sending separate messages."

export const FILE_TOO_LARGE_MESSAGE = "could not upload file, maximum input length exceeded"

export const IMAGE_UNSUPPORTED_MODEL_MESSAGE = "Selected model does not support image uploads"

export const STORAGE_KEYS = {
  CLIENT_PROFILE: "client-profile",
  DRAFT_PREFIX: "draft-",
} as const

export const FEEDBACK_KEY = "ux.thumb_vote" as const

