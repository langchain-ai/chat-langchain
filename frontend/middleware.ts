import { auth } from "@/lib/config/auth-config"
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export default auth((request) => {
  const { pathname } = request.nextUrl
  // Check if auth is required based on deployment environment
  const deploymentEnv = process.env.NEXT_PUBLIC_DEPLOYMENT_ENV
  const requiresAuth = deploymentEnv === "internal"

  // Allow access to API routes and static files
  if (
    pathname.startsWith('/api/auth') ||
    pathname.startsWith('/_next') ||
    pathname.startsWith('/favicon.ico')
  ) {
    return NextResponse.next()
  }

  // If auth is not required (external deployment), allow all access
  if (!requiresAuth) {
    return NextResponse.next()
  }

  // If auth is required but user is not authenticated, allow access
  // The sign-in modal on the page will handle authentication
  // (No redirect needed - modal appears on the chat page)
  return NextResponse.next()
})

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
