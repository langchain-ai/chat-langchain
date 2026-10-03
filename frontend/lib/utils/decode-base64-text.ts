export function decodeBase64Text(base64: string, mimeType?: string): string {
  const bytes = Uint8Array.from(atob(base64 || ""), character =>
    character.charCodeAt(0)
  )
  const charset = mimeType?.match(/charset\s*=\s*["']?([^;"'\s]+)/i)?.[1] ?? "utf-8"

  try {
    return new TextDecoder(charset, { fatal: false }).decode(bytes)
  } catch {
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes)
  }
}
