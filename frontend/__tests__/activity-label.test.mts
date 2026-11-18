import assert from "node:assert/strict"
import test from "node:test"
import { activityLabel } from "../lib/utils/chat/activity-label.ts"

test("while streaming the block shows the latest step in place", () => {
  assert.equal(activityLabel(["Searching docs"], true), "Searching docs…")
  assert.equal(activityLabel(["Searching docs", "Reading Persistence page."], true), "Reading Persistence page… (step 2)")
})

test("once settled it summarises the steps", () => {
  assert.equal(activityLabel(["a", "b", "c"], false), "Agent steps (3)")
})
