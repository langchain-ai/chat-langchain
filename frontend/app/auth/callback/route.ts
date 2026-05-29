import { createServerClient } from "@supabase/ssr"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get("code")
  const oauthError = requestUrl.searchParams.get("error")
  const oauthErrorDescription = requestUrl.searchParams.get("error_description")
  const origin = requestUrl.origin

  if (oauthError) {
    console.error("OAuth provider returned an error", {
      hasError: Boolean(oauthError),
      hasDescription: Boolean(oauthErrorDescription),
    })
    const redirectUrl = new URL(origin)
    redirectUrl.searchParams.set("auth_error", "oauth_failed")
    return NextResponse.redirect(redirectUrl)
  }

  if (!code) {
    console.error("OAuth callback missing authorization code")
    return NextResponse.redirect(`${origin}?auth_error=missing_code`)
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  if (!url || !key) {
    console.error("Supabase credentials are not configured")
    return NextResponse.redirect(`${origin}?auth_error=auth_not_configured`)
  }

  try {
    const cookieStore = await cookies()
    const supabase = createServerClient(url, key, {
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
      const redirectUrl = new URL(origin)
      redirectUrl.searchParams.set("auth_error", "auth_failed")
      return NextResponse.redirect(redirectUrl)
    }

    return NextResponse.redirect(origin)
  } catch (error) {
    console.error("Unexpected error in OAuth callback", {
      isError: error instanceof Error,
    })
    return NextResponse.redirect(`${origin}?auth_error=unexpected_error`)
  }
}
