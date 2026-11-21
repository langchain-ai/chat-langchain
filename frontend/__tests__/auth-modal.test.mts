import assert from "node:assert/strict"
import test from "node:test"
import { isComposerTypingTarget, shouldAutoOpenAuthModal } from "../lib/utils/auth-modal.ts"

const base = { enabled: true, loading: false, signedIn: false, dismissed: false, userStartedTyping: false }

test("opens for a signed-out visitor who hasn't started typing", () => {
  assert.equal(shouldAutoOpenAuthModal(base), true)
})

test("never interrupts a visitor who already started writing", () => {
  assert.equal(shouldAutoOpenAuthModal({ ...base, userStartedTyping: true }), false)
})

test("stays closed while loading, when signed in, dismissed or disabled", () => {
  assert.equal(shouldAutoOpenAuthModal({ ...base, loading: true }), false)
  assert.equal(shouldAutoOpenAuthModal({ ...base, signedIn: true }), false)
  assert.equal(shouldAutoOpenAuthModal({ ...base, dismissed: true }), false)
  assert.equal(shouldAutoOpenAuthModal({ ...base, enabled: false }), false)
})

test("counts the composer textarea, not inputs inside a dialog", () => {
  assert.equal(isComposerTypingTarget({ tagName: "TEXTAREA", closest: () => null }), true)
  assert.equal(isComposerTypingTarget({ tagName: "TEXTAREA", closest: () => ({}) }), false)
  assert.equal(isComposerTypingTarget({ tagName: "INPUT", closest: () => null }), false)
  assert.equal(isComposerTypingTarget(null), false)
})
