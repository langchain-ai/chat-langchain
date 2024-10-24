// src/middleware.ts
import createMiddleware from 'next-intl/middleware';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { routing } from './i18n/routing';

// 创建一个函数来检查认证
function checkAuth(request: NextRequest) {
  const token = request.cookies.get('token')?.value;

  // 检查访问 rich-master-ai 路径的请求
  if (request.nextUrl.pathname.includes('/components/RichMasterAI')) {
    if (!token) {
      // 获取当前语言
      const locale = request.nextUrl.pathname.split('/')[1] || 'zh';
      // 重定向到对应语言的登录页面
      return NextResponse.redirect(new URL(`/${locale}/login`, request.url));
    }
  }
  return null;
}

// 合并 next-intl middleware 和认证检查
const intlMiddleware = createMiddleware(routing);

export default async function middleware(request: NextRequest) {
  // 首先检查认证
  const authResult = checkAuth(request);
  if (authResult) {
    return authResult;
  }

  // 如果认证通过或不需要认证，继续处理国际化
  return intlMiddleware(request);
}

// 更新 matcher 配置以包含所有需要的路径
export const config = {
  matcher: [
    // 国际化路径
    '/',
    '/(zh|en)/:path*',
    // 需要保护的路径
    '/components/RichMasterAI/:path*',
    '/(zh|en)/components/RichMasterAI/:path*'
  ]
};