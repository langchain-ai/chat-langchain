import { cookies } from "next/headers"
import { NextResponse } from "next/server"

const COOKIE_NAME = "chat_langchain_guest"

// Public docs-agent deployment. Guest issuance is a managed, public route
// (POST /identity/guest) served by the MDA deployment.
const DEPLOYMENT_URL =
  process.env.NEXT_PUBLIC_LANGGRAPH_API_URL ||
  process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL ||
  (process.env.NODE_ENV === "development" ? "http://127.0.0.1:2024" : undefined)

export const runtime = "nodejs"

interface GuestClaims {
  sub?: string
  exp?: number
}

function decodeJwtClaims(token: string): GuestClaims | null {
  const parts = token.split(".")
  if (parts.length !== 3 || !parts[1]) return null
  try {
    const normalized = parts[1].replaceAll("-", "+").replaceAll("_", "/")
    const padding = "=".repeat((4 - (normalized.length % 4)) % 4)
    const json = Buffer.from(`${normalized}${padding}`, "base64").toString("utf8")
    return JSON.parse(json) as GuestClaims
  } catch {
    return null
  }
}

function isFresh(claims: GuestClaims | null): claims is GuestClaims & { sub: string } {
  if (!claims || typeof claims.sub !== "string") return false
  if (typeof claims.exp !== "number") return true
  // Refresh a minute early so callers never receive a token that expires mid-use.
  return claims.exp - 60 > Date.now() / 1000
}

async function issueManagedGuestToken(): Promise<string> {
  if (!DEPLOYMENT_URL) {
    throw new Error("NEXT_PUBLIC_LANGGRAPH_API_URL is not configured")
  }

  const headers: Record<string, string> = { "Content-Type": "application/json" }
  const authKey = process.env.NEXT_PUBLIC_LANGGRAPH_AUTH_KEY
  if (authKey) headers["X-Auth-Key"] = authKey

  const response = await fetch(
    `${DEPLOYMENT_URL.replace(/\/$/, "")}/identity/guest`,
    { method: "POST", headers }
  )
  if (!response.ok) {
    throw new Error(`Guest issuance failed with status ${response.status}`)
  }

  const data = (await response.json()) as { token?: string }
  if (!data.token) {
    throw new Error("Guest issuance response was missing a token")
  }
  return data.token
}

export async function POST() {
  const cookieStore = await cookies()

  // Reuse the cached managed token while it is still valid so a guest keeps the
  // same identity (and thread history) across reloads within the token's TTL.
  const cached = cookieStore.get(COOKIE_NAME)?.value
  const cachedClaims = cached ? decodeJwtClaims(cached) : null
  if (cached && isFresh(cachedClaims)) {
    return NextResponse.json({ guestId: cachedClaims.sub, token: cached })
  }

  let token: string
  try {
    token = await issueManagedGuestToken()
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Guest issuance failed" },
      { status: 502 }
    )
  }

  const claims = decodeJwtClaims(token)
  if (!claims?.sub) {
    return NextResponse.json(
      { error: "Guest token was missing a subject claim" },
      { status: 502 }
    )
  }

  const maxAge = claims.exp
    ? Math.max(0, Math.floor(claims.exp - Date.now() / 1000))
    : undefined
  const response = NextResponse.json({ guestId: claims.sub, token })
  response.cookies.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    ...(maxAge !== undefined ? { maxAge } : {}),
  })
  return response
}
