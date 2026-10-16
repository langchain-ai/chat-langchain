import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import {
  decodeGuestSession,
  encodeGuestSession,
  guestSessionCookieName,
  parseGuestAuthRegion,
  type GuestAuthRegion,
} from "@/lib/auth/guest-session"

const GUEST_SESSION_MAX_AGE_SECONDS = 30 * 24 * 60 * 60

const supabaseConfigs: Record<
  GuestAuthRegion,
  { url?: string; anonKey?: string }
> = {
  us: {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  },
  eu: {
    url: process.env.NEXT_PUBLIC_SUPABASE_EU_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_EU_ANON_KEY,
  },
  apac: {
    url: process.env.NEXT_PUBLIC_SUPABASE_APAC_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_APAC_ANON_KEY,
  },
  aws: {
    url: process.env.NEXT_PUBLIC_SUPABASE_AWS_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_AWS_ANON_KEY,
  },
}

export const runtime = "nodejs"

class GuestIssuanceError extends Error {
  constructor(readonly status: number) {
    super("Guest authentication failed")
  }
}

function guestClient(region: GuestAuthRegion): SupabaseClient | null {
  const { url, anonKey } = supabaseConfigs[region]
  if (!url || !anonKey) return null

  return createClient(url, anonKey, {
    auth: {
      autoRefreshToken: false,
      detectSessionInUrl: false,
      persistSession: false,
    },
  })
}

async function verifiedAnonymousSession(
  client: SupabaseClient,
  session: Session
): Promise<{ guestId: string; session: Session } | null> {
  const { data, error } = await client.auth.getUser(session.access_token)
  if (error || !data.user?.is_anonymous) return null
  return { guestId: data.user.id, session }
}

async function restoreGuestSession(
  client: SupabaseClient,
  cookieValue: string | undefined
): Promise<{ guestId: string; session: Session } | null> {
  if (!cookieValue) return null
  const stored = decodeGuestSession(cookieValue)
  if (!stored) return null

  const { data, error } = await client.auth.setSession({
    access_token: stored.accessToken,
    refresh_token: stored.refreshToken,
  })
  if (error || !data.session) return null
  return verifiedAnonymousSession(client, data.session)
}

async function issueGuestSession(
  client: SupabaseClient
): Promise<{ guestId: string; session: Session }> {
  const { data, error } = await client.auth.signInAnonymously()
  if (error) {
    throw new GuestIssuanceError(error.status === 429 ? 429 : 502)
  }
  if (!data.session || !data.user?.is_anonymous) {
    throw new GuestIssuanceError(502)
  }
  return { guestId: data.user.id, session: data.session }
}

export async function POST(request: Request) {
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 })
  }

  const region = parseGuestAuthRegion(
    body && typeof body === "object" ? (body as { region?: unknown }).region : null
  )
  if (!region) {
    return NextResponse.json({ error: "Invalid auth region" }, { status: 400 })
  }

  const client = guestClient(region)
  if (!client) {
    return NextResponse.json(
      { error: "Guest authentication is unavailable" },
      { status: 503 }
    )
  }

  const cookieStore = await cookies()
  const cookieName = guestSessionCookieName(region)
  let guest = await restoreGuestSession(
    client,
    cookieStore.get(cookieName)?.value
  )

  if (!guest) {
    try {
      guest = await issueGuestSession(client)
    } catch (error) {
      const status = error instanceof GuestIssuanceError ? error.status : 502
      return NextResponse.json(
        { error: "Guest authentication failed" },
        { status }
      )
    }
  }

  const response = NextResponse.json({
    guestId: guest.guestId,
    token: guest.session.access_token,
  })
  response.cookies.set(
    cookieName,
    encodeGuestSession({
      accessToken: guest.session.access_token,
      refreshToken: guest.session.refresh_token,
    }),
    {
      httpOnly: true,
      maxAge: GUEST_SESSION_MAX_AGE_SECONDS,
      path: "/",
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
    }
  )
  return response
}
