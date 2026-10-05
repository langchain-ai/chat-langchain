import assert from "node:assert/strict"
import test from "node:test"
import { isImeComposing, shouldSubmitOnEnter } from "../lib/ime.ts"

const event = (key: string, shiftKey: boolean, isComposing = false, keyCode?: number) => ({
  key,
  shiftKey,
  nativeEvent: { isComposing },
  keyCode,
})

test("ignores Enter while an IME composition is active", () => {
  assert.equal(isImeComposing(event("Enter", false, true), false), true)
  assert.equal(isImeComposing(event("Enter", false, false, 229), false), true)
  assert.equal(shouldSubmitOnEnter(event("Enter", false, true), false), false)
  assert.equal(shouldSubmitOnEnter(event("Enter", false, false, 229), false), false)
})

test("does not treat ordinary Enter or Shift+Enter as composition", () => {
  assert.equal(shouldSubmitOnEnter(event("Enter", false, false, 13), false), true)
  assert.equal(shouldSubmitOnEnter(event("Enter", true, false, 13), false), false)
})
