import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Utf8 } from "../lib/utils/file-decoding.ts"

test("decodes UTF-8 attachment content without corruption", () => {
  const content = "# 你好，世界\n\nCafé — 👋"
  const base64 = Buffer.from(content, "utf8").toString("base64")

  assert.equal(decodeBase64Utf8(base64), content)
})

test("rejects invalid UTF-8 attachment content", () => {
  const invalidUtf8 = Buffer.from([0xff, 0xfe]).toString("base64")

  assert.throws(() => decodeBase64Utf8(invalidUtf8), TypeError)
})
