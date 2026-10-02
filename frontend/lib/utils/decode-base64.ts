export function decodeBase64Text(base64: string): string {
  const binaryString = atob(base64)
  const bytes = Uint8Array.from(binaryString, character => character.charCodeAt(0))
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes)
}
