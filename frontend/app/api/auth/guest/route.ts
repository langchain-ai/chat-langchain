import { createHmac, randomUUID, timingSafeEqual } from "node:crypto"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"

const COOKIE_NAME = "chat_langchain_guest"
const TOKEN_PREFIX = "guest"
const TOKEN_TTL_SECONDS = 60 * 60
const COOKIE_TTL_SECONDS = 60 * 60 * 24 * 30

export const runtime = "nodejs"

function getSecret(): string {
  const secret = process.env.GUEST_AUTH_SECRET
  if (!secret) {
    throw new Error("GUEST_AUTH_SECRET is not configured")
  }
  return secret
}

function base64UrlEncode(value: Buffer | string): string {
  return Buffer.from(value)
    .toString("base64")
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "")
}

function base64UrlDecode(value: string): Buffer {
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/")
  const padding = "=".repeat((4 - (normalized.length % 4)) % 4)
  return Buffer.from(`${normalized}${padding}`, "base64")
}

function signPayload(payload: string): string {
  return base64UrlEncode(createHmac("sha256", getSecret()).update(payload).digest())
}

function issueGuestToken(guestId: string, ttlSeconds: number): string {
  const now = Math.floor(Date.now() / 1000)
  const payload = base64UrlEncode(
    JSON.stringify({
      typ: "guest",
      sub: guestId,
      iat: now,
      exp: now + ttlSeconds,
    })
  )
  return `${TOKEN_PREFIX}.${payload}.${signPayload(payload)}`
}

function verifyGuestToken(token: string): string | null {
  const [prefix, payload, signature] = token.split(".")
  if (prefix !== TOKEN_PREFIX || !payload || !signature) return null

  const expected = signPayload(payload)
  const expectedBuffer = Buffer.from(expected)
  const actualBuffer = Buffer.from(signature)
  if (
    expectedBuffer.length !== actualBuffer.length ||
    !timingSafeEqual(expectedBuffer, actualBuffer)
  ) {
    return null
  }

  try {
    const decoded = JSON.parse(base64UrlDecode(payload).toString("utf8"))
    if (decoded?.typ !== "guest") return null
    if (typeof decoded?.exp !== "number" || decoded.exp < Date.now() / 1000) {
      return null
    }
    if (typeof decoded?.sub !== "string" || !decoded.sub.startsWith("user-")) {
      return null
    }
    return decoded.sub
  } catch {
    return null
  }
}

export async function POST() {
  let guestId: string | null = null
  const cookieStore = await cookies()
  const existingToken = cookieStore.get(COOKIE_NAME)?.value
  if (existingToken) {
    guestId = verifyGuestToken(existingToken)
  }

  if (!guestId) {
    guestId = `user-${randomUUID()}`
  }

  const cookieToken = issueGuestToken(guestId, COOKIE_TTL_SECONDS)
  const bearerToken = issueGuestToken(guestId, TOKEN_TTL_SECONDS)
  const response = NextResponse.json({ guestId, token: bearerToken })
  response.cookies.set(COOKIE_NAME, cookieToken, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: COOKIE_TTL_SECONDS,
  })
  return response
}
