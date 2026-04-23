import { NextRequest, NextResponse } from "next/server"

import {
  createOrUpdateFeedback,
  deleteFeedback,
  isLangSmithConfigError,
  isLangSmithNotFoundError,
} from "@/lib/server/langsmith"

export const runtime = "nodejs"

interface FeedbackRequest {
  runId: string
  feedbackKey: string
  score: "positive" | "negative"
  comment?: string
  feedbackId?: string
}

export async function POST(request: NextRequest) {
  let payload: FeedbackRequest

  try {
    payload = await request.json()
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 })
  }

  try {
    const result = await createOrUpdateFeedback(payload)
    return NextResponse.json(result)
  } catch (error) {
    if (isLangSmithConfigError(error)) {
      return NextResponse.json(
        { error: "LangSmith tracing is not configured for this deployment" },
        { status: 503 }
      )
    }

    if (isLangSmithNotFoundError(error)) {
      return NextResponse.json({ error: "Run or feedback not found" }, { status: 404 })
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to submit feedback" },
      { status: 500 }
    )
  }
}

export async function DELETE(request: NextRequest) {
  const feedbackId = request.nextUrl.searchParams.get("feedbackId")

  if (!feedbackId) {
    return NextResponse.json({ error: "feedbackId is required" }, { status: 400 })
  }

  try {
    await deleteFeedback(feedbackId)
    return NextResponse.json({ success: true })
  } catch (error) {
    if (isLangSmithConfigError(error)) {
      return NextResponse.json(
        { error: "LangSmith tracing is not configured for this deployment" },
        { status: 503 }
      )
    }

    if (isLangSmithNotFoundError(error)) {
      return NextResponse.json({ error: "Feedback not found" }, { status: 404 })
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to delete feedback" },
      { status: 500 }
    )
  }
}
