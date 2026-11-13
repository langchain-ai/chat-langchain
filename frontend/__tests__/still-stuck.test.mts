import assert from "node:assert/strict"
import test from "node:test"
import { FORUM_URL, SUPPORT_URL, stillStuckLinks } from "../lib/utils/chat/still-stuck.ts"

test("prefills the forum topic title with the question", () => {
  const links = stillStuckLinks("How do I add a checkpointer?")
  assert.equal(links.forum, `${FORUM_URL}new-topic?title=How%20do%20I%20add%20a%20checkpointer%3F`)
  assert.equal(links.support, SUPPORT_URL)
})

test("drops the page-context note, collapses whitespace and caps the length", () => {
  const q = "why   is\nmy bill higher\n\n---\nContext about the user's current page: ..."
  assert.equal(stillStuckLinks(q).forum, `${FORUM_URL}new-topic?title=why%20is%20my%20bill%20higher`)
  const long = "x".repeat(500)
  const title = decodeURIComponent(new URL(stillStuckLinks(long).forum).searchParams.get("title") ?? "")
  assert.equal(title.length, 120)
})

test("falls back to the plain forum link without a question", () => {
  assert.equal(stillStuckLinks("").forum, FORUM_URL)
  assert.equal(stillStuckLinks(undefined).forum, FORUM_URL)
})
