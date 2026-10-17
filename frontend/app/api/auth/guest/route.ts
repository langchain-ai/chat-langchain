import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import {
  newGuestSubject,
  readGuestToken,
  signGuestToken,
} from "@/lib/server/guest-token"

const COOKIE_NAME = "chat_langchain_guest"
const GUEST_SESSION_MAX_AGE_SECONDS = 24 * 60 * 60

export const runtime = "nodejs"

function guestSecret(): string | null {
  const secret = process.env.GUEST_AUTH_SECRET
  return secret || null
}

export async function POST() {
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
  const token =
    existing && cached ? cached : signGuestToken(secret, newGuestSubject())
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
