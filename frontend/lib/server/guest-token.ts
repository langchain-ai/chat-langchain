import { createHmac, randomUUID, timingSafeEqual } from "node:crypto"

export const GUEST_TOKEN_ISSUER = "mda:guest"
const TTL_SECONDS = 24 * 60 * 60

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
): { sub: string } | null {
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
    return { sub: claims.sub }
  } catch {
    return null
  }
}
