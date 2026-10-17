import assert from "node:assert/strict"
import test from "node:test"
import {
  decodeGuestSession,
  encodeGuestSession,
  guestSessionCookieName,
  parseGuestAuthRegion,
} from "../lib/auth/guest-session.ts"

test("accepts only configured auth regions", () => {
  assert.equal(parseGuestAuthRegion("us"), "us")
  assert.equal(parseGuestAuthRegion("eu"), "eu")
  assert.equal(parseGuestAuthRegion("unknown"), null)
  assert.equal(parseGuestAuthRegion({ region: "us" }), null)
})

test("isolates guest cookies by region", () => {
  assert.equal(guestSessionCookieName("us"), "chat_langchain_guest_us")
  assert.equal(guestSessionCookieName("eu"), "chat_langchain_guest_eu")
  assert.notEqual(guestSessionCookieName("us"), guestSessionCookieName("eu"))
})

test("round trips a stored guest session", () => {
  const stored = {
    accessToken: "access-token",
    refreshToken: "refresh-token",
  }

  assert.deepEqual(decodeGuestSession(encodeGuestSession(stored)), stored)
})

test("rejects malformed guest session cookies", () => {
  assert.equal(decodeGuestSession("not-base64-json"), null)
  assert.equal(
    decodeGuestSession(
      Buffer.from(JSON.stringify({ accessToken: "only-one" })).toString(
        "base64url"
      )
    ),
    null
  )
  assert.equal(decodeGuestSession("x".repeat(8193)), null)
})
