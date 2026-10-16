import assert from "node:assert/strict"
import test from "node:test"
import {
  newGuestSubject,
  readGuestToken,
  signGuestToken,
} from "../lib/server/guest-token.ts"

const SECRET = "test-guest-secret"

test("round trips a guest token", () => {
  const subject = newGuestSubject()
  const token = signGuestToken(SECRET, subject)

  assert.equal(readGuestToken(SECRET, token)?.sub, subject)
})

test("rejects a token signed with a different secret", () => {
  const token = signGuestToken(SECRET, newGuestSubject())

  assert.equal(readGuestToken("other-secret", token), null)
})

test("rejects an expired guest token", () => {
  const token = signGuestToken(SECRET, newGuestSubject(), 0)

  assert.equal(readGuestToken(SECRET, token, 25 * 60 * 60 * 1000), null)
})
