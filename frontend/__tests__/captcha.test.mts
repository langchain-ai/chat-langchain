import assert from "node:assert/strict"
import test from "node:test"
import {
  CAPTCHA_LOAD_FAILED_MESSAGE,
  getCaptchaErrorMessage,
  isCaptchaLoadError,
} from "../lib/auth/captcha.ts"

test("treats only script failures as load errors", () => {
  assert.equal(isCaptchaLoadError("script-error"), true)
  assert.equal(isCaptchaLoadError("script-load-error-other"), true)
  assert.equal(isCaptchaLoadError("network-error"), false)
  assert.equal(isCaptchaLoadError("challenge-closed"), false)
})

test("maps hCaptcha string codes to user-facing messages", () => {
  assert.equal(
    getCaptchaErrorMessage("challenge-closed"),
    "The security check was closed before it finished. Please try again."
  )
  assert.equal(
    getCaptchaErrorMessage("rate-limited"),
    "Too many attempts. Please wait a moment and try again."
  )
  assert.equal(
    getCaptchaErrorMessage("script-error"),
    CAPTCHA_LOAD_FAILED_MESSAGE
  )
})

test("shows nothing when the widget cancels a pending check", () => {
  for (const code of [
    "react-component-unmounted",
    "hcaptcha-reset",
    "hcaptcha-removed",
    "hcaptcha-closed",
    "hcaptcha-execute-replaced",
  ]) {
    assert.equal(getCaptchaErrorMessage(new Error(code)), null)
  }
})

test("never returns raw error text", () => {
  const generic =
    "Something went wrong with the security check. Please try again."
  assert.equal(
    getCaptchaErrorMessage(
      new TypeError("Cannot read properties of undefined (reading 'execute')")
    ),
    generic
  )
  assert.equal(getCaptchaErrorMessage("invalid-data"), generic)
  assert.equal(getCaptchaErrorMessage("constructor"), generic)
  assert.equal(getCaptchaErrorMessage(undefined), generic)
})
