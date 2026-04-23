import { NextRequest, NextResponse } from "next/server"

import {
  isLangSmithConfigError,
  isLangSmithNotFoundError,
  readRun,
} from "@/lib/server/langsmith"

export const runtime = "nodejs"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params

  try {
    const run = await readRun(runId)
    return NextResponse.json(run)
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
      { error: error instanceof Error ? error.message : "Failed to fetch run" },
      { status: 500 }
    )
  }
}
