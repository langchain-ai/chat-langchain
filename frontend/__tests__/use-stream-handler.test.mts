import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Utf8 } from "../lib/utils/decode-base64.ts"

test("decodes base64-encoded UTF-8 text without mojibake", () => {
  const original = "你好，世界 🌍"
  const bytes = new TextEncoder().encode(original)
  const base64 = btoa(String.fromCharCode(...bytes))

  assert.equal(decodeBase64Utf8(base64), original)
})
