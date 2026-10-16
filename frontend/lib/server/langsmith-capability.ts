const FEEDBACK_CAPABILITY = "langsmith:chat-feedback"
const TRACE_CAPABILITY = "langsmith:trace-viewer"
const FEEDBACK_KEY = "ux.thumb_vote"
const MAX_BODY_CHARS = 8_192
const MAX_COMMENT_CHARS = 2_000
const THREAD_ACCESS_TIMEOUT_MS = 5_000

type JsonRecord = Record<string, unknown>

export interface LangSmithCapabilityClient {
  readRun(runId: string): Promise<unknown>
  shareRun(runId: string): Promise<string>
  createFeedback(
    runId: string,
    key: string,
    options: {
      score: number
      value: string
      comment?: string
      sourceInfo: { mda_actor_id: string }
      feedbackSourceType: "api"
    }
  ): Promise<{ id: string }>
  updateFeedback(
    feedbackId: string,
    options: { score: number; value: string; comment?: string }
  ): Promise<void>
  readFeedback(feedbackId: string): Promise<unknown>
  deleteFeedback(feedbackId: string): Promise<void>
  listFeedback(options: {
    runIds: string[]
    feedbackKeys: string[]
  }): AsyncIterable<unknown>
}

export interface LangSmithCapabilityDependencies {
  client: LangSmithCapabilityClient
  fetch: typeof fetch
  langgraphUrl: string
}

class CapabilityError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function json(data: JsonRecord, status = 200): Response {
  return Response.json(data, { status })
}

async function requestBody(request: Request): Promise<JsonRecord> {
  const raw = await request.text()
  if (raw.length > MAX_BODY_CHARS) {
    throw new CapabilityError(413, "Request body is too large")
  }

  try {
    const parsed: unknown = JSON.parse(raw)
    if (!isRecord(parsed)) throw new Error("not an object")
    return parsed
  } catch {
    throw new CapabilityError(400, "Invalid request")
  }
}

function requiredString(
  value: unknown,
  field: string,
  maxLength = 256
): string {
  if (typeof value !== "string" || !value || value.length > maxLength) {
    throw new CapabilityError(400, `Invalid ${field}`)
  }
  return value
}

function feedbackInput(body: JsonRecord): {
  score: number
  value: "positive" | "negative"
  comment?: string
} {
  const score = body.score
  if (score !== 0 && score !== 1) {
    throw new CapabilityError(400, "Invalid feedback score")
  }

  const value = score === 1 ? "positive" : "negative"
  if (body.value !== value) {
    throw new CapabilityError(400, "Invalid feedback value")
  }

  const comment = body.comment
  if (comment !== undefined && comment !== null) {
    if (typeof comment !== "string" || comment.length > MAX_COMMENT_CHARS) {
      throw new CapabilityError(400, "Invalid feedback comment")
    }
    return { score, value, comment }
  }
  return { score, value }
}

function threadUrl(baseUrl: string, threadId: string): string {
  let url: URL
  try {
    url = new URL(baseUrl)
  } catch {
    throw new CapabilityError(503, "Thread verification is unavailable")
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password
  ) {
    throw new CapabilityError(503, "Thread verification is unavailable")
  }

  url.pathname = `${url.pathname.replace(/\/$/, "")}/threads/${encodeURIComponent(threadId)}`
  url.search = ""
  url.hash = ""
  return url.toString()
}

async function verifyThreadAccess(
  request: Request,
  threadId: string,
  dependencies: LangSmithCapabilityDependencies
): Promise<string> {
  const authorization = request.headers.get("authorization")
  if (!authorization?.startsWith("Bearer ") || authorization.length <= 7) {
    throw new CapabilityError(401, "Unauthorized")
  }

  const region = request.headers.get("x-supabase-region")
  if (!region || !["us", "eu", "apac", "aws"].includes(region)) {
    throw new CapabilityError(401, "Unauthorized")
  }

  const headers = new Headers({
    Authorization: authorization,
    "X-Supabase-Region": region,
  })

  let response: Response
  try {
    response = await dependencies.fetch(
      threadUrl(dependencies.langgraphUrl, threadId),
      {
        cache: "no-store",
        headers,
        redirect: "error",
        signal: AbortSignal.timeout(THREAD_ACCESS_TIMEOUT_MS),
      }
    )
  } catch {
    throw new CapabilityError(502, "Failed to verify thread access")
  }

  if (response.status === 401) throw new CapabilityError(401, "Unauthorized")
  if (response.status === 403 || response.status === 404) {
    throw new CapabilityError(403, "Thread is not accessible")
  }
  if (!response.ok) {
    throw new CapabilityError(502, "Failed to verify thread access")
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new CapabilityError(502, "Failed to verify thread access")
  }
  const metadata = isRecord(payload) ? payload.metadata : null
  const owner = isRecord(metadata) ? metadata.owner : null
  if (typeof owner !== "string" || !owner) {
    throw new CapabilityError(403, "Thread is not accessible")
  }
  return owner
}

export function runThreadId(run: unknown): string | null {
  if (!isRecord(run)) return null
  for (const field of ["thread_id", "threadId"]) {
    const value = run[field]
    if (typeof value === "string" && value) return value
  }

  const metadataCandidates: unknown[] = [run.metadata]
  if (isRecord(run.extra)) {
    metadataCandidates.push(
      isRecord(run.extra.metadata) ? run.extra.metadata : run.extra
    )
  }
  for (const metadata of metadataCandidates) {
    if (!isRecord(metadata)) continue
    for (const field of ["thread_id", "threadId", "session_id", "sessionId"]) {
      const value = metadata[field]
      if (typeof value === "string" && value) return value
    }
  }
  return null
}

async function scopedRun(
  client: LangSmithCapabilityClient,
  runId: string,
  threadId: string
): Promise<JsonRecord> {
  const run = await client.readRun(runId)
  if (!isRecord(run) || runThreadId(run) !== threadId) {
    throw new CapabilityError(403, "Run does not belong to thread")
  }
  return run
}

const RUN_RESPONSE_FIELDS = [
  "id",
  "status",
  "start_time",
  "end_time",
  "url",
  "metadata",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "prompt_cost",
  "completion_cost",
  "total_cost",
] as const

export function shapeRun(run: JsonRecord): JsonRecord {
  return Object.fromEntries(
    RUN_RESPONSE_FIELDS.flatMap((field) =>
      run[field] === undefined ? [] : [[field, run[field]]]
    )
  )
}

function feedbackKey(feedback: unknown): string | null {
  if (!isRecord(feedback) || typeof feedback.key !== "string") return null
  return feedback.key
}

function feedbackActorId(feedback: unknown): string | null {
  if (!isRecord(feedback) || !isRecord(feedback.feedback_source)) return null
  const metadata = feedback.feedback_source.metadata
  if (!isRecord(metadata)) return null
  return typeof metadata.mda_actor_id === "string"
    ? metadata.mda_actor_id
    : null
}

function feedbackRunId(feedback: unknown): string {
  if (!isRecord(feedback)) throw new CapabilityError(404, "Feedback not found")
  return requiredString(feedback.run_id, "feedback", 128)
}

function assertOwnedThumbFeedback(feedback: unknown, owner: string): void {
  if (
    feedbackKey(feedback) !== FEEDBACK_KEY ||
    feedbackActorId(feedback) !== owner
  ) {
    throw new CapabilityError(403, "Feedback is not accessible")
  }
}

async function handleTraceCapability(
  body: JsonRecord,
  threadId: string,
  client: LangSmithCapabilityClient
): Promise<JsonRecord> {
  const action = body.action
  if (action !== "read" && action !== "share") {
    throw new CapabilityError(403, "Action is not allowed")
  }
  const runId = requiredString(body.run_id, "run", 128)
  const run = await scopedRun(client, runId, threadId)
  if (action === "read") return shapeRun(run)
  return { url: await client.shareRun(runId) }
}

async function handleFeedbackCapability(
  body: JsonRecord,
  threadId: string,
  owner: string,
  client: LangSmithCapabilityClient
): Promise<JsonRecord> {
  const action = body.action
  if (!["create", "update", "delete"].includes(String(action))) {
    throw new CapabilityError(403, "Action is not allowed")
  }

  if (action === "create") {
    if (body.key !== FEEDBACK_KEY) {
      throw new CapabilityError(400, "Invalid feedback key")
    }
    const runId = requiredString(body.run_id, "run", 128)
    await scopedRun(client, runId, threadId)
    const input = feedbackInput(body)
    for await (const feedback of client.listFeedback({
      runIds: [runId],
      feedbackKeys: [FEEDBACK_KEY],
    })) {
      if (feedbackActorId(feedback) === owner) {
        throw new CapabilityError(409, "Feedback already exists")
      }
    }
    const feedback = await client.createFeedback(runId, FEEDBACK_KEY, {
      ...input,
      sourceInfo: { mda_actor_id: owner },
      feedbackSourceType: "api",
    })
    return { id: feedback.id }
  }

  const feedbackId = requiredString(body.feedback_id, "feedback", 128)
  const feedback = await client.readFeedback(feedbackId)
  await scopedRun(client, feedbackRunId(feedback), threadId)
  assertOwnedThumbFeedback(feedback, owner)
  if (action === "delete") {
    await client.deleteFeedback(feedbackId)
    return {}
  }

  await client.updateFeedback(feedbackId, feedbackInput(body))
  return { id: feedbackId }
}

function upstreamStatus(error: unknown): number | null {
  if (!isRecord(error)) return null
  for (const key of ["status", "status_code", "statusCode"]) {
    const value = error[key]
    if (typeof value === "number") return value
  }
  return isRecord(error.response) && typeof error.response.status === "number"
    ? error.response.status
    : null
}

export async function handleLangSmithCapability(
  request: Request,
  capabilityId: string,
  dependencies: LangSmithCapabilityDependencies
): Promise<Response> {
  try {
    if (![FEEDBACK_CAPABILITY, TRACE_CAPABILITY].includes(capabilityId)) {
      throw new CapabilityError(404, "Unknown capability")
    }
    const body = await requestBody(request)
    const threadId = requiredString(body.thread_id, "thread")
    const owner = await verifyThreadAccess(request, threadId, dependencies)
    const data =
      capabilityId === TRACE_CAPABILITY
        ? await handleTraceCapability(body, threadId, dependencies.client)
        : await handleFeedbackCapability(
            body,
            threadId,
            owner,
            dependencies.client
          )
    return json({ data })
  } catch (error) {
    if (error instanceof CapabilityError) {
      return json({ error: error.message }, error.status)
    }
    const status = upstreamStatus(error) === 404 ? 404 : 502
    return json(
      { error: status === 404 ? "Resource not found" : "LangSmith request failed" },
      status
    )
  }
}
