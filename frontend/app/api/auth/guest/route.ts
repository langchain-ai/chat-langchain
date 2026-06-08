import { createHmac, randomUUID, timingSafeEqual } from "node:crypto"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const COOKIE_NAME = "chat_langchain_guest"
const TOKEN_PREFIX = "guest"
const TOKEN_TTL_SECONDS = 60 * 60
const COOKIE_TTL_SECONDS = 60 * 60 * 24 * 30
const GUEST_AUTH_MAX_REQUESTS = Number(
  process.env.GUEST_AUTH_RATE_LIMIT_MAX_REQUESTS ?? 10
)
const GUEST_AUTH_WINDOW_MS = Number(
  process.env.GUEST_AUTH_RATE_LIMIT_WINDOW_MS ?? 60_000
)
const GUEST_AUTH_COOLDOWN_MS = Number(
  process.env.GUEST_AUTH_RATE_LIMIT_COOLDOWN_MS ?? 60_000
)

export const runtime = "nodejs"

interface RateLimitEntry {
  timestamps: number[]
  blockedUntil: number
}

const rateLimitEntries = new Map<string, RateLimitEntry>()

function getClientIp(request: NextRequest): string {
  const forwardedFor = request.headers.get("x-forwarded-for")
  if (forwardedFor) return forwardedFor.split(",")[0]?.trim() || "unknown"
  return request.headers.get("x-real-ip") || "unknown"
}

function checkGuestAuthRateLimit(request: NextRequest): NextResponse | null {
  const now = Date.now()
  const ip = getClientIp(request)
  let entry = rateLimitEntries.get(ip)

  if (!entry) {
    entry = { timestamps: [], blockedUntil: 0 }
    rateLimitEntries.set(ip, entry)
  }

  if (entry.blockedUntil > now) {
    const retryAfter = Math.ceil((entry.blockedUntil - now) / 1000)
    return rateLimitBlockedResponse(retryAfter)
  }

  if (entry.blockedUntil > 0) {
    entry.blockedUntil = 0
    entry.timestamps = []
  }

  const cutoff = now - GUEST_AUTH_WINDOW_MS
  entry.timestamps = entry.timestamps.filter((timestamp) => timestamp > cutoff)

  if (entry.timestamps.length >= GUEST_AUTH_MAX_REQUESTS) {
    entry.blockedUntil = now + GUEST_AUTH_COOLDOWN_MS
    return rateLimitBlockedResponse(Math.ceil(GUEST_AUTH_COOLDOWN_MS / 1000))
  }

  entry.timestamps.push(now)
  return null
}

function rateLimitBlockedResponse(retryAfter: number): NextResponse {
  return NextResponse.json(
    {
      error: "Too many requests. Please slow down and try again later.",
      retry_after: retryAfter,
    },
    {
      status: 429,
      headers: {
        "Retry-After": String(retryAfter),
        "X-RateLimit-Limit": String(GUEST_AUTH_MAX_REQUESTS),
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": String(retryAfter),
      },
    }
  )
}

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

export async function POST(request: NextRequest) {
  const rateLimitResponse = checkGuestAuthRateLimit(request)
  if (rateLimitResponse) return rateLimitResponse

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
