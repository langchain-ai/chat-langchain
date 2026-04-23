"use client";

import { useEffect } from "react";
import { useUserId } from "@/lib/hooks/auth/use-user-id";

// Declare analytics global type
declare global {
  interface Window {
    analytics: any;
  }
}

/**
 * Segment Analytics Provider
 *
 * Leverages existing langgraph-user-id for tracking anonymous users.
 */
export function SegmentProvider({ children }: { children: React.ReactNode }) {
  const userId = useUserId(); // Uses existing langgraph-user-id logic

  useEffect(() => {
    // Wait for Segment to load and user ID to be ready
    if (typeof window === "undefined" || !window.analytics || !userId) {
      return;
    }

    window.analytics.identify(userId, {
      deployment: "public",
      userType: "anonymous",
    });

    window.analytics.page({
      deployment: "public",
    });

  }, [userId]);

  return <>{children}</>;
}

/**
 * Track custom events for the public app.
 */
export function trackEvent(eventName: string, properties?: Record<string, any>): void {
  if (typeof window === "undefined" || !window.analytics) {
    return;
  }

  window.analytics.track(eventName, {
    ...properties,
    deployment: "public",
  });
}
