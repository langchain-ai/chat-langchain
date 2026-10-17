export const GUEST_AUTH_REGIONS = ["us", "eu", "apac", "aws"] as const

export type GuestAuthRegion = (typeof GUEST_AUTH_REGIONS)[number]

export interface StoredGuestSession {
  accessToken: string
  refreshToken: string
}

export function parseGuestAuthRegion(value: unknown): GuestAuthRegion | null {
  return typeof value === "string" &&
    GUEST_AUTH_REGIONS.includes(value as GuestAuthRegion)
    ? (value as GuestAuthRegion)
    : null
}

export function guestSessionCookieName(region: GuestAuthRegion): string {
  return `chat_langchain_guest_${region}`
}

export function encodeGuestSession(session: StoredGuestSession): string {
  return Buffer.from(JSON.stringify(session), "utf8").toString("base64url")
}

export function decodeGuestSession(value: string): StoredGuestSession | null {
  if (!value || value.length > 8192) return null

  try {
    const parsed = JSON.parse(
      Buffer.from(value, "base64url").toString("utf8")
    ) as Partial<StoredGuestSession>
    if (
      typeof parsed.accessToken !== "string" ||
      !parsed.accessToken ||
      typeof parsed.refreshToken !== "string" ||
      !parsed.refreshToken
    ) {
      return null
    }
    return {
      accessToken: parsed.accessToken,
      refreshToken: parsed.refreshToken,
    }
  } catch {
    return null
  }
}
