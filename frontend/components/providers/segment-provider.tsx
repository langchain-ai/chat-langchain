"use client";

import { useEffect, useRef } from "react";
import { useAuth } from "@/lib/auth";

// Declare analytics global type
declare global {
  interface Window {
    analytics: any;
  }
}

/**
 * Segment Analytics Provider
 *
 * Identifies signed-in users by their real Supabase auth user ID (matching
 * the pattern used in langchain-ai/help-portal), not by email or a
 * client-generated guest ID. Anonymous visitors are left to Segment's own
 * anonymous ID rather than being identify()'d with a fabricated ID.
 */
export function SegmentProvider({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const identifiedUserIdRef = useRef<string | null>(null);

  // Identify the user once we have a real auth user ID.
  useEffect(() => {
    if (typeof window === "undefined" || !window.analytics) return;
    if (loading || !user) return;
    if (identifiedUserIdRef.current === user.id) return;

    identifiedUserIdRef.current = user.id;
    window.analytics.identify(user.id, {
      email: user.email,
      name: user.user_metadata?.full_name,
      deployment: "public",
      userType: "authenticated",
    });
  }, [user, loading])

  // Reset identity on logout so the next sign-in starts clean.
  useEffect(() => {
    if (typeof window === "undefined" || !window.analytics) return;
    if (loading || user) return
    if (identifiedUserIdRef.current === null) return

    identifiedUserIdRef.current = null
    window.analytics.reset()
  }, [user, loading])

  // Track page views regardless of auth state.
  useEffect(() => {
    if (typeof window === "undefined" || !window.analytics || loading) return;

    window.analytics.page({
      deployment: "public",
    });
  }, [loading]);

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
