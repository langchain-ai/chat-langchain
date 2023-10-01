// JS backend not used by default, see README for instructions.

import { NextRequest, NextResponse } from "next/server";

import { Client } from "langsmith";

export const runtime = "edge";

const client = new Client();

const pollForRun = async (runId: string, retryCount = 0): Promise<string> => {
  await new Promise((resolve) =>
    setTimeout(resolve, retryCount * retryCount * 100),
  );
  try {
    await client.readRun(runId);
  } catch (e) {
    return pollForRun(runId, retryCount + 1);
  }
  try {
    const sharedLink = await client.readRunSharedLink(runId);
    if (!sharedLink) {
      throw new Error("Run is not shared.");
    }
    return sharedLink;
  } catch (e) {
    return client.shareRun(runId);
  }
};

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { run_id } = body;
    if (run_id === undefined) {
      return NextResponse.json(
        { error: "No run ID provided" },
        { status: 400 },
      );
    }
    const response = await pollForRun(run_id);
    return NextResponse.json(response, { status: 200 });
  } catch (e: any) {
    console.log(e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
