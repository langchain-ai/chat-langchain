import assert from "node:assert/strict"
import test from "node:test"
import { shouldIgnoreCompositionKeyDown } from "../lib/hooks/use-composition-guard.ts"

const enterEvent = (overrides: Partial<{ isComposing: boolean; keyCode: number }> = {}) => ({
  nativeEvent: { isComposing: false, ...overrides },
  keyCode: overrides.keyCode,
})

test("ignores Enter while the native event is composing", () => {
  assert.equal(shouldIgnoreCompositionKeyDown(enterEvent({ isComposing: true }), false), true)
})

test("ignores Enter with the IME key code", () => {
  assert.equal(shouldIgnoreCompositionKeyDown(enterEvent({ keyCode: 229 }), false), true)
})

test("ignores Safari's post-composition commit keydown", () => {
  assert.equal(shouldIgnoreCompositionKeyDown(enterEvent(), true), true)
})

test("allows normal Enter after composition ends", () => {
  assert.equal(shouldIgnoreCompositionKeyDown(enterEvent(), false), false)
})
