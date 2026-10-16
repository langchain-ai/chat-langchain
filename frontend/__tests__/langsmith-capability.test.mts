import assert from "node:assert/strict"
import test from "node:test"
import {
  handleLangSmithCapability,
  type LangSmithCapabilityClient,
} from "../lib/server/langsmith-capability.ts"

const THREAD_ID = "00000000-0000-0000-0000-000000000001"
const RUN_ID = "00000000-0000-0000-0000-000000000002"
const OWNER_ID = "00000000-0000-0000-0000-000000000003"

function request(body: Record<string, unknown>, authenticated = true): Request {
  return new Request("http://localhost/api/langsmith", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(authenticated
        ? {
            Authorization: "Bearer test-token",
            "X-Supabase-Region": "us",
          }
        : {}),
    },
    body: JSON.stringify(body),
  })
}

function fakeClient(overrides: Partial<LangSmithCapabilityClient> = {}) {
  const client: LangSmithCapabilityClient = {
    async readRun(runId) {
      return {
        id: runId,
        inputs: { private: true },
        outputs: { private: true },
        extra: { metadata: { thread_id: THREAD_ID } },
        prompt_tokens: 10,
        total_tokens: 20,
      }
    },
    async shareRun() {
      return "https://smith.langchain.com/public/example"
    },
    async createFeedback() {
      return { id: "feedback-id" }
    },
    async updateFeedback() {},
    async readFeedback() {
      return { run_id: RUN_ID }
    },
    async deleteFeedback() {},
    async *listFeedback() {},
    ...overrides,
  }
  return client
}

function dependencies(
  client: LangSmithCapabilityClient,
  threadResponse = new Response(
    JSON.stringify({ metadata: { owner: OWNER_ID } }),
    { status: 200 }
  )
) {
  return {
    client,
    fetch: (async () => threadResponse.clone()) as typeof fetch,
    langgraphUrl: "https://example.langgraph.app",
  }
}

test("requires caller authentication before reading LangSmith", async () => {
  let readRunCalled = false
  const client = fakeClient({
    async readRun() {
      readRunCalled = true
      return {}
    },
  })

  const response = await handleLangSmithCapability(
    request(
      { action: "read", run_id: RUN_ID, thread_id: THREAD_ID },
      false
    ),
    "langsmith:trace-viewer",
    dependencies(client)
  )

  assert.equal(response.status, 401)
  assert.equal(readRunCalled, false)
})

test("rejects a run from a different thread", async () => {
  const client = fakeClient({
    async readRun(runId) {
      return {
        id: runId,
        extra: { metadata: { thread_id: "another-thread" } },
      }
    },
  })

  const response = await handleLangSmithCapability(
    request({ action: "read", run_id: RUN_ID, thread_id: THREAD_ID }),
    "langsmith:trace-viewer",
    dependencies(client)
  )

  assert.equal(response.status, 403)
})

test("returns only allowlisted run fields", async () => {
  const response = await handleLangSmithCapability(
    request({ action: "read", run_id: RUN_ID, thread_id: THREAD_ID }),
    "langsmith:trace-viewer",
    dependencies(fakeClient())
  )
  const payload = (await response.json()) as {
    data: Record<string, unknown>
  }

  assert.equal(response.status, 200)
  assert.deepEqual(payload.data, {
    id: RUN_ID,
    prompt_tokens: 10,
    total_tokens: 20,
  })
  assert.equal("inputs" in payload.data, false)
  assert.equal("outputs" in payload.data, false)
  assert.equal("extra" in payload.data, false)
})

test("prevents duplicate feedback from the same verified owner", async () => {
  const client = fakeClient({
    async *listFeedback() {
      yield {
        feedback_source: {
          metadata: { mda_actor_id: OWNER_ID },
        },
      }
    },
  })

  const response = await handleLangSmithCapability(
    request({
      action: "create",
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      key: "ux.thumb_vote",
      score: 1,
      value: "positive",
    }),
    "langsmith:chat-feedback",
    dependencies(client)
  )

  assert.equal(response.status, 409)
})

test("stamps new feedback with the verified thread owner", async () => {
  let sourceInfo: { mda_actor_id: string } | undefined
  const client = fakeClient({
    async createFeedback(_runId, _key, options) {
      sourceInfo = options.sourceInfo
      return { id: "feedback-id" }
    },
  })

  const response = await handleLangSmithCapability(
    request({
      action: "create",
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      key: "ux.thumb_vote",
      score: 0,
      value: "negative",
    }),
    "langsmith:chat-feedback",
    dependencies(client)
  )
  const payload = (await response.json()) as {
    data: Record<string, unknown>
  }

  assert.equal(response.status, 200)
  assert.deepEqual(sourceInfo, { mda_actor_id: OWNER_ID })
  assert.equal(payload.data.id, "feedback-id")
})

test("does not update feedback attached to another thread", async () => {
  let updated = false
  const client = fakeClient({
    async readRun(runId) {
      return {
        id: runId,
        extra: { metadata: { thread_id: "another-thread" } },
      }
    },
    async updateFeedback() {
      updated = true
    },
  })

  const response = await handleLangSmithCapability(
    request({
      action: "update",
      feedback_id: "feedback-id",
      thread_id: THREAD_ID,
      score: 1,
      value: "positive",
    }),
    "langsmith:chat-feedback",
    dependencies(client)
  )

  assert.equal(response.status, 403)
  assert.equal(updated, false)
})
