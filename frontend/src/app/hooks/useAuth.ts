// hooks/useAuth.ts
import { useEffect } from 'react';
import { useRouter } from 'next/router';

export function useAuth(redirectTo: string = '/login') {
  const router = useRouter();

  useEffect(() => {
    const isAuthenticated = false; // 在这里替换成实际的登录状态检查

    if (!isAuthenticated) {
      router.replace(redirectTo);
    }
  }, [router, redirectTo]);

  return { /* 在这里可以返回用户信息或其他状态 */ };
}