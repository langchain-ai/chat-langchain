import assert from "node:assert/strict"
import test from "node:test"
import { decodeTextAttachment } from "../lib/utils/chat/content-helpers.ts"

test("text attachments retain UTF-8 content when sent or rerun", () => {
  const content = "# Café\n안녕하세요 — LangChain 🦜\n"
  assert.equal(decodeTextAttachment(Buffer.from(content).toString("base64")), content)
})

test("invalid base64 is rejected instead of submitting corrupted file content", () => {
  assert.throws(() => decodeTextAttachment("%%%"))
})
