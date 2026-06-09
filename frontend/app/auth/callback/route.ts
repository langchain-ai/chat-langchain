import { createServerClient } from "@supabase/ssr"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

type AuthRegion = "us" | "eu" | "apac" | "aws"
const SUPABASE_AUTH_STORAGE_KEY_PREFIX = "chat-langchain-supabase-auth"

function isAuthRegion(value: string | null): value is AuthRegion {
  return value === "us" || value === "eu" || value === "apac" || value === "aws"
}

function getSupabaseCredentials(region: AuthRegion): {
  url?: string
  key?: string
} {
  if (region === "eu") {
    return {
      url: process.env.NEXT_PUBLIC_SUPABASE_EU_URL,
      key: process.env.NEXT_PUBLIC_SUPABASE_EU_ANON_KEY,
    }
  }
  if (region === "apac") {
    return {
      url: process.env.NEXT_PUBLIC_SUPABASE_APAC_URL,
      key: process.env.NEXT_PUBLIC_SUPABASE_APAC_ANON_KEY,
    }
  }
  if (region === "aws") {
    return {
      url: process.env.NEXT_PUBLIC_SUPABASE_AWS_URL,
      key: process.env.NEXT_PUBLIC_SUPABASE_AWS_ANON_KEY,
    }
  }
  return {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL,
    key: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  }
}

function redirectWithAuthError(origin: string, errorCode: string): NextResponse {
  const redirectUrl = new URL(origin)
  redirectUrl.searchParams.set("auth_error", errorCode)
  return NextResponse.redirect(redirectUrl)
}

function getSupabaseAuthStorageKey(region: AuthRegion): string {
  return `${SUPABASE_AUTH_STORAGE_KEY_PREFIX}-${region}`
}

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get("code")
  const oauthError = requestUrl.searchParams.get("error")
  const oauthErrorDescription = requestUrl.searchParams.get("error_description")
  const origin = requestUrl.origin
  const requestedRegion = requestUrl.searchParams.get("region")
  const region: AuthRegion = isAuthRegion(requestedRegion) ? requestedRegion : "us"

  if (oauthError) {
    console.error("OAuth provider returned an error", {
      hasError: Boolean(oauthError),
      hasDescription: Boolean(oauthErrorDescription),
    })
    return redirectWithAuthError(origin, "oauth_failed")
  }

  if (!code) {
    console.error("OAuth callback missing authorization code")
    return redirectWithAuthError(origin, "missing_code")
  }

  const { url, key } = getSupabaseCredentials(region)
  if (!url || !key) {
    console.error("Supabase credentials are not configured", { region })
    return redirectWithAuthError(origin, "auth_not_configured")
  }

  try {
    const cookieStore = await cookies()
    const supabase = createServerClient(url, key, {
      auth: {
        storageKey: getSupabaseAuthStorageKey(region),
      },
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            )
          } catch {
            // This can be called from a Server Component in some render paths.
          }
        },
      },
    })

    const { error } = await supabase.auth.exchangeCodeForSession(code)
    if (error) {
      console.error("Failed to exchange code for session", {
        hasMessage: Boolean(error.message),
      })
      return redirectWithAuthError(origin, "auth_failed")
    }

    return NextResponse.redirect(origin)
  } catch (error) {
    console.error("Unexpected error in OAuth callback", {
      isError: error instanceof Error,
    })
    return redirectWithAuthError(origin, "unexpected_error")
  }
}
