"use client";

import { SessionProvider as NextAuthSessionProvider } from "next-auth/react";

export function SessionProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextAuthSessionProvider
      // Reduce session polling - only check every 15 minutes (in seconds)
      refetchInterval={15 * 60} // 900 seconds = 15 minutes
      // Disable refetch on window focus to reduce API calls
      // With 6+ components using useSession, this causes excessive polling
      refetchOnWindowFocus={false}
    >
      {children}
    </NextAuthSessionProvider>
  );
}
