export const CAPTCHA_LOADING_MESSAGE =
  "The security check is still loading. Please try again in a moment."

export const CAPTCHA_LOAD_FAILED_MESSAGE =
  "Couldn't load the security check. A firewall or browser extension may be blocking it. Please try again."

const GENERIC_CAPTCHA_MESSAGE =
  "Something went wrong with the security check. Please try again."

const CAPTCHA_MESSAGES = new Map([
  [
    "challenge-closed",
    "The security check was closed before it finished. Please try again.",
  ],
  ["challenge-expired", "The security check expired. Please try again."],
  [
    "network-error",
    "Couldn't reach the security check. Check your connection and try again.",
  ],
  ["rate-limited", "Too many attempts. Please wait a moment and try again."],
  ["script-error", CAPTCHA_LOAD_FAILED_MESSAGE],
  ["script-load-error-other", CAPTCHA_LOAD_FAILED_MESSAGE],
])

// Raised when the widget is torn down or reset while a check is pending, for
// example when the modal closes. Nothing failed from the user's point of view.
const CANCELLED_CODES = new Set([
  "react-component-unmounted",
  "hcaptcha-reset",
  "hcaptcha-removed",
  "hcaptcha-closed",
  "hcaptcha-execute-replaced",
])

export const isCaptchaLoadError = (code: string): boolean =>
  code === "script-error" || code === "script-load-error-other"

/**
 * Maps a rejected hCaptcha execute() to text for the sign-in form, or null when
 * there is nothing to show. hCaptcha rejects with string codes and the React
 * wrapper rejects with Errors. Raw values are never returned.
 */
export function getCaptchaErrorMessage(error: unknown): string | null {
  const code = error instanceof Error ? error.message : String(error)
  if (CANCELLED_CODES.has(code)) return null
  return CAPTCHA_MESSAGES.get(code) ?? GENERIC_CAPTCHA_MESSAGE
}
