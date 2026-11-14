import assert from "node:assert/strict"
import test from "node:test"
import { absolutizeDocsLinks } from "../lib/utils/chat/content-helpers.ts"

test("site-relative docs links point at docs.langchain.com", () => {
  assert.equal(
    absolutizeDocsLinks("See [Persistence](/oss/python/langgraph/persistence) and [Deploy](/langsmith/deployment#cloud)."),
    "See [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) and [Deploy](https://docs.langchain.com/langsmith/deployment#cloud)."
  )
})

test("absolute links, other relative paths and code fences are unchanged", () => {
  const md = "[a](https://docs.langchain.com/oss/x) [b](/pricing)\n```md\n[c](/oss/y)\n```\n[d](/oss/z)"
  assert.equal(
    absolutizeDocsLinks(md),
    "[a](https://docs.langchain.com/oss/x) [b](/pricing)\n```md\n[c](/oss/y)\n```\n[d](https://docs.langchain.com/oss/z)"
  )
})
