import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

function getCorsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': '*',
  };
}


async function handleRequest(req: NextRequest, method: string) {

  const path = req.nextUrl.pathname.replace(/^\/?api\//, "");
  
  // 处理登录请求
  if (path === 'auth/login' && method === 'POST') {
    try {
      const body = await req.json();
      const { email, password } = body;

      // 示例认证逻辑
      if (email === 'test@example.com' && password === '123456') {
        return NextResponse.json({ 
          success: true, 
          token: 'secure-auth-token' 
        }, {
          headers: getCorsHeaders()
        });
      }

      return NextResponse.json({ 
        success: false, 
        message: '邮箱或密码错误' 
      }, { 
        status: 401,
        headers: getCorsHeaders()
      });
    } catch (error) {
      return NextResponse.json({ 
        success: false, 
        message: '服务器错误' 
      }, { 
        status: 500,
        headers: getCorsHeaders()
      });
    }
  }


  try {
    const path = req.nextUrl.pathname.replace(/^\/?api\//, "");
    const url = new URL(req.url);
    const searchParams = new URLSearchParams(url.search);
    searchParams.delete("_path");
    searchParams.delete("nxtP_path");
    const queryString = searchParams.toString()
      ? `?${searchParams.toString()}`
      : "";

    const options: RequestInit = {
      method,
      headers: {
        "x-api-key": process.env.LANGCHAIN_API_KEY || "",
      },
    };

    if (["POST", "PUT", "PATCH"].includes(method)) {
      options.body = await req.text();
    }

    const res = await fetch(
      `${process.env.API_BASE_URL}/${path}${queryString}`,
      options,
    );

    return new NextResponse(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: {
        ...res.headers,
        ...getCorsHeaders(),
      },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}


export const GET = (req: NextRequest) => handleRequest(req, "GET");
export const POST = (req: NextRequest) => handleRequest(req, "POST");
export const PUT = (req: NextRequest) => handleRequest(req, "PUT");
export const PATCH = (req: NextRequest) => handleRequest(req, "PATCH");
export const DELETE = (req: NextRequest) => handleRequest(req, "DELETE");

// Add a new OPTIONS handler
export const OPTIONS = () => {
  return new NextResponse(null, {
    status: 204,
    headers: {
      ...getCorsHeaders(),
    },
  });
};
