import { NextRequest, NextResponse } from "next/server"

import {
  getTraceUrl,
  isLangSmithConfigError,
  isLangSmithNotFoundError,
} from "@/lib/server/langsmith"

export const runtime = "nodejs"

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params

  try {
    const shareUrl = await getTraceUrl(runId)
    return NextResponse.json({ shareUrl })
  } catch (error) {
    if (isLangSmithConfigError(error)) {
      return NextResponse.json(
        { error: "LangSmith tracing is not configured for this deployment" },
        { status: 503 }
      )
    }

    if (isLangSmithNotFoundError(error)) {
      return NextResponse.json({ error: "Run not found yet" }, { status: 404 })
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to share run" },
      { status: 500 }
    )
  }
}
