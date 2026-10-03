export function decodeBase64Utf8(base64: string): string {
  const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0))
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes)
}
