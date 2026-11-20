import assert from "node:assert/strict"
import test from "node:test"
import { isNewFinalMessage, lastAiMessageId } from "../lib/utils/chat/final-message.ts"

test("lastAiMessageId finds the previous turn's answer", () => {
  assert.equal(lastAiMessageId([{ type: "human", id: "h1" }, { type: "ai", id: "a1" }, { type: "human", id: "h2" }]), "a1")
  assert.equal(lastAiMessageId([{ type: "human", id: "h1" }]), null)
})

test("an AI message different from the baseline is new", () => {
  assert.equal(isNewFinalMessage({ captured: true, id: "a1" }, { id: "a2" }), true)
  assert.equal(isNewFinalMessage({ captured: true, id: null }, { id: "a1" }), true)
})

test("the previous answer, a missing id, or no baseline is not new", () => {
  assert.equal(isNewFinalMessage({ captured: true, id: "a1" }, { id: "a1" }), false)
  assert.equal(isNewFinalMessage({ captured: true, id: "a1" }, {}), false)
  assert.equal(isNewFinalMessage({ captured: false, id: null }, { id: "a2" }), false)
})
