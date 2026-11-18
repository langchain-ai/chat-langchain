import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import {
  newGuestSubject,
  readGuestToken,
  requestClientIp,
  signForwardedIp,
  signGuestToken,
} from "@/lib/server/guest-token"

const COOKIE_NAME = "chat_langchain_guest"
const GUEST_SESSION_MAX_AGE_SECONDS = 24 * 60 * 60

export const runtime = "nodejs"

function guestSecret(): string | null {
  const secret = process.env.GUEST_AUTH_SECRET
  return secret || null
}

function engineUrl(): string | null {
  const url =
    process.env.NEXT_PUBLIC_LANGGRAPH_API_URL ||
    process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL
  return url ? url.replace(/\/+$/, "") : null
}

type EngineMint =
  | { kind: "token"; token: string }
  | { kind: "limited"; retryAfter: string | null }
  | { kind: "unavailable" }

/**
 * Mint new passes at the engine so its per-IP cap (rate_limit.py) covers this
 * route too. The visitor's IP goes along signed with the shared guest secret;
 * otherwise every mint would share Vercel's egress IPs.
 */
async function mintAtEngine(secret: string, request: Request): Promise<EngineMint> {
  const base = engineUrl()
  if (!base) return { kind: "unavailable" }
  const headers: Record<string, string> = { "Content-Type": "application/json" }
  const ip = requestClientIp(request.headers)
  if (ip) {
    headers["x-guest-client-ip"] = ip
    headers["x-guest-client-sig"] = signForwardedIp(secret, ip)
  }
  try {
    const res = await fetch(`${base}/identity/guest`, {
      method: "POST",
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    })
    if (res.status === 429) {
      return { kind: "limited", retryAfter: res.headers.get("retry-after") }
    }
    if (!res.ok) return { kind: "unavailable" }
    const data = (await res.json()) as { token?: unknown }
    if (typeof data.token !== "string" || !readGuestToken(secret, data.token)) {
      return { kind: "unavailable" }
    }
    return { kind: "token", token: data.token }
  } catch {
    return { kind: "unavailable" }
  }
}

export async function POST(request: Request) {
  const secret = guestSecret()
  if (!secret) {
    return NextResponse.json(
      { error: "Guest authentication is unavailable" },
      { status: 503 }
    )
  }

  const cookieStore = await cookies()
  const cached = cookieStore.get(COOKIE_NAME)?.value
  const existing = cached ? readGuestToken(secret, cached) : null
  let token: string
  if (existing && cached) {
    token = cached
  } else {
    const minted = await mintAtEngine(secret, request)
    if (minted.kind === "limited") {
      const limited = NextResponse.json(
        { error: "Too many guest sessions. Try again later." },
        { status: 429 }
      )
      if (minted.retryAfter) limited.headers.set("Retry-After", minted.retryAfter)
      return limited
    }
    // Engine down or not reachable: chat won't work either, but keep the old
    // local mint so a blip there doesn't block guests from loading the page.
    token =
      minted.kind === "token"
        ? minted.token
        : signGuestToken(secret, newGuestSubject())
  }
  const guestId = readGuestToken(secret, token)?.sub
  if (!guestId) {
    return NextResponse.json(
      { error: "Guest authentication failed" },
      { status: 502 }
    )
  }

  const response = NextResponse.json({ guestId, token })
  response.cookies.set(COOKIE_NAME, token, {
    httpOnly: true,
    maxAge: GUEST_SESSION_MAX_AGE_SECONDS,
    path: "/",
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
  })
  return response
}
