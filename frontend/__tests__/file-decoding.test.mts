import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Text } from "../lib/utils/chat/decode-base64-text.ts"

test("decodes UTF-8 attachment text without corrupting CJK characters", () => {
  const original = "你好，世界！这是一个测试文件。"
  const base64 = Buffer.from(original, "utf8").toString("base64")

  assert.equal(decodeBase64Text(base64), original)
})
