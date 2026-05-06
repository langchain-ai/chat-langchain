import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'

const SITE_PASSWORD = process.env.SITE_PASSWORD || 'changeme'
const AUTH_COOKIE_NAME = 'site-auth'
const AUTH_TOKEN = 'authenticated'

export async function POST(request: NextRequest) {
  try {
    const { password } = await request.json()

    if (password === SITE_PASSWORD) {
      const cookieStore = await cookies()

      const deploymentEnv = process.env.NEXT_PUBLIC_DEPLOYMENT_ENV

      // Set cookie that expires in 7 days
      // For internal deployment, use sameSite=none to support iframe embedding
      cookieStore.set(AUTH_COOKIE_NAME, AUTH_TOKEN, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: deploymentEnv === 'internal' ? 'none' : 'strict',
        maxAge: 60 * 60 * 24 * 7, // 7 days
        path: '/',
      })

      return NextResponse.json({ success: true })
    }

    return NextResponse.json({ success: false }, { status: 401 })
  } catch (error) {
    return NextResponse.json({ success: false }, { status: 500 })
  }
}
