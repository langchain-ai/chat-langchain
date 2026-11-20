import { createHmac, randomUUID, timingSafeEqual } from "node:crypto"

export const GUEST_TOKEN_ISSUER = "mda:guest"
const TTL_SECONDS = 24 * 60 * 60
/** Re-sign a still-valid pass for the same guest once it has less than this left. */
export const RENEW_BELOW_SECONDS = 12 * 60 * 60

function encode(value: string): string {
  return Buffer.from(value).toString("base64url")
}

export function newGuestSubject(): string {
  return `guest:${randomUUID()}`
}

export function signGuestToken(
  secret: string,
  subject: string,
  now = Date.now()
): string {
  const header = encode(JSON.stringify({ alg: "HS256", typ: "JWT" }))
  const issued = Math.floor(now / 1000)
  const body = encode(
    JSON.stringify({
      iss: GUEST_TOKEN_ISSUER,
      sub: subject,
      iat: issued,
      exp: issued + TTL_SECONDS,
    })
  )
  const signature = createHmac("sha256", secret)
    .update(`${header}.${body}`)
    .digest("base64url")
  return `${header}.${body}.${signature}`
}

export function readGuestToken(
  secret: string,
  token: string,
  now = Date.now()
): { sub: string; exp: number } | null {
  const parts = token.split(".")
  if (parts.length !== 3) return null
  const [header, body, signature] = parts
  const expected = createHmac("sha256", secret)
    .update(`${header}.${body}`)
    .digest("base64url")
  const actualBuf = Buffer.from(signature)
  const expectedBuf = Buffer.from(expected)
  if (
    actualBuf.length !== expectedBuf.length ||
    !timingSafeEqual(actualBuf, expectedBuf)
  ) {
    return null
  }

  try {
    const claims = JSON.parse(
      Buffer.from(body, "base64url").toString("utf8")
    ) as { iss?: unknown; sub?: unknown; exp?: unknown }
    if (claims.iss !== GUEST_TOKEN_ISSUER) return null
    if (typeof claims.sub !== "string" || !claims.sub.startsWith("guest:")) {
      return null
    }
    if (typeof claims.exp !== "number" || claims.exp * 1000 <= now) return null
    return { sub: claims.sub, exp: claims.exp }
  } catch {
    return null
  }
}

/**
 * Sliding renewal: a valid pass close to expiry gets a fresh 24h for the SAME
 * guest id, so returning guests keep their (owner-scoped) thread history
 * instead of silently becoming a new guest 24h after their first visit.
 * Returns null when the pass is invalid or doesn't need renewing yet.
 */
export function renewGuestToken(
  secret: string,
  token: string,
  now = Date.now()
): string | null {
  const claims = readGuestToken(secret, token, now)
  if (!claims) return null
  if (claims.exp * 1000 - now > RENEW_BELOW_SECONDS * 1000) return null
  return signGuestToken(secret, claims.sub, now)
}

/**
 * Signature the engine checks before trusting a forwarded client IP for its
 * per-IP guest mint cap (rate_limit.forwarded_client_ip). Format "<ts>.<hex>".
 */
export function signForwardedIp(
  secret: string,
  ip: string,
  now = Date.now()
): string {
  const ts = Math.floor(now / 1000)
  const mac = createHmac("sha256", secret).update(`${ip}.${ts}`).digest("hex")
  return `${ts}.${mac}`
}

/** Visitor IP as Vercel reports it (Vercel overwrites these headers). */
export function requestClientIp(headers: Headers): string | null {
  const real = headers.get("x-real-ip")?.trim()
  if (real) return real
  const first = headers.get("x-forwarded-for")?.split(",")[0]?.trim()
  return first || null
}
