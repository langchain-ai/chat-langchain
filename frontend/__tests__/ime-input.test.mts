import assert from "node:assert/strict"
import test from "node:test"
import { shouldSubmitOnEnter } from "../lib/utils/keyboard.ts"

function keyEvent(overrides: Partial<Parameters<typeof shouldSubmitOnEnter>[0]> = {}) {
  return {
    key: "Enter",
    keyCode: 13,
    shiftKey: false,
    nativeEvent: {},
    ...overrides,
  }
}

test("does not submit an Enter keydown while an IME is composing", () => {
  assert.equal(
    shouldSubmitOnEnter(keyEvent({ nativeEvent: { isComposing: true } })),
    false
  )
  assert.equal(shouldSubmitOnEnter(keyEvent({ keyCode: 229 })), false)
})

test("submits a normal Enter keydown but not Shift+Enter", () => {
  assert.equal(shouldSubmitOnEnter(keyEvent()), true)
  assert.equal(shouldSubmitOnEnter(keyEvent({ shiftKey: true })), false)
})
