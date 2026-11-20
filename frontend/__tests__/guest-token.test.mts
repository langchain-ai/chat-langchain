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

test("signs a forwarded client IP the engine can check", async () => {
  const { signForwardedIp } = await import("../lib/server/guest-token.ts")
  const { createHmac } = await import("node:crypto")
  const sig = signForwardedIp(SECRET, "34.71.41.140", 1_700_000_000_000)
  const [ts, mac] = sig.split(".")
  assert.equal(ts, "1700000000")
  assert.equal(
    mac,
    createHmac("sha256", SECRET).update("34.71.41.140.1700000000").digest("hex")
  )
})

test("reads the visitor IP from Vercel headers", async () => {
  const { requestClientIp } = await import("../lib/server/guest-token.ts")
  assert.equal(requestClientIp(new Headers({ "x-real-ip": "5.6.7.8" })), "5.6.7.8")
  assert.equal(
    requestClientIp(new Headers({ "x-forwarded-for": "1.2.3.4, 10.0.0.1" })),
    "1.2.3.4"
  )
  assert.equal(requestClientIp(new Headers()), null)
})

test("renews a pass close to expiry for the same guest", async () => {
  const { renewGuestToken } = await import("../lib/server/guest-token.ts")
  const subject = newGuestSubject()
  const token = signGuestToken(SECRET, subject, 0)
  // 13h in: more than 12h left, nothing to renew.
  assert.equal(renewGuestToken(SECRET, token, 11 * 60 * 60 * 1000), null)
  // 20h in: renewed, same subject, expiry pushed 24h from now.
  const now = 20 * 60 * 60 * 1000
  const renewed = renewGuestToken(SECRET, token, now)
  assert.ok(renewed)
  const claims = readGuestToken(SECRET, renewed, now)
  assert.equal(claims?.sub, subject)
  assert.equal(claims?.exp, now / 1000 + 24 * 60 * 60)
  // Still valid past the original expiry.
  assert.equal(readGuestToken(SECRET, renewed, 30 * 60 * 60 * 1000)?.sub, subject)
})

test("never renews an expired or forged pass", async () => {
  const { renewGuestToken } = await import("../lib/server/guest-token.ts")
  const token = signGuestToken(SECRET, newGuestSubject(), 0)
  assert.equal(renewGuestToken(SECRET, token, 25 * 60 * 60 * 1000), null)
  assert.equal(renewGuestToken("other-secret", token, 20 * 60 * 60 * 1000), null)
})
