import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Utf8 } from "../lib/utils/decode-base64.ts"

test("decodes UTF-8 text with CJK and accented Latin characters", () => {
  const text = "# eKDP — 企业知识 café"
  const base64 = Buffer.from(text, "utf8").toString("base64")

  assert.equal(decodeBase64Utf8(base64), text)
})

test("leaves ASCII text unchanged", () => {
  const text = "# README\nPlain ASCII content"
  const base64 = Buffer.from(text, "utf8").toString("base64")

  assert.equal(decodeBase64Utf8(base64), text)
})
