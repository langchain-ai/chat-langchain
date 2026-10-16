import { GUEST_TOKEN_ISSUER, readGuestToken } from "@/lib/server/guest-token"

export const runtime = "nodejs"

export async function GET(request: Request) {
  const secret = process.env.GUEST_AUTH_SECRET
  if (!secret) {
    return Response.json(
      { error: "Guest authentication is unavailable" },
      { status: 503 }
    )
  }

  const authorization = request.headers.get("authorization") || ""
  const token = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length).trim()
    : ""
  const claims = token ? readGuestToken(secret, token) : null
  if (!claims) {
    return Response.json({ error: "Unauthorized" }, { status: 401 })
  }

  return Response.json({ iss: GUEST_TOKEN_ISSUER, sub: claims.sub })
}
