export function decodeBase64Utf8(b64: string): string {
  const bytes = Uint8Array.from(atob(b64), character => character.charCodeAt(0))
  return new TextDecoder("utf-8").decode(bytes)
}
