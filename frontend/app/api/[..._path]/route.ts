import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  try {
    const body = await req.text();
    const path = req.nextUrl.pathname.replace(/^\/?api\//, "");
    // Remove Next dynamic routing param
    const url = new URL(req.url);
    const searchParams = new URLSearchParams(url.search);
    searchParams.delete("_path");
    const queryString = searchParams.toString()
      ? `?${searchParams.toString()}`
      : "";
    const res = await fetch(
      `${process.env.API_BASE_URL}/${path}${queryString}`,
      {
        method: "POST",
        body,
        headers: req.headers,
      },
    );
    return new NextResponse(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}

export async function GET(req: NextRequest) {
  try {
    const path = req.nextUrl.pathname.replace(/^\/?api\//, "");
    // Remove Next dynamic routing param
    const url = new URL(req.url);
    const searchParams = new URLSearchParams(url.search);
    searchParams.delete("_path");
    const queryString = searchParams.toString()
      ? `?${searchParams.toString()}`
      : "";
    const res = await fetch(
      `${process.env.API_BASE_URL}/${path}${queryString}`,
      {
        method: "GET",
        headers: req.headers,
      },
    );
    return new NextResponse(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
