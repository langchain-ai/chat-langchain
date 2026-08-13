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
  //
  // We can't rely solely on identifiedUserIdRef here: it's component-local
  // and starts out null on every reload. If a session expires or is revoked
  // between visits, the next load resolves straight to user === null while
  // the ref is still null too, which would skip reset() and leave Segment's
  // own persisted identity (in localStorage) attributed to the former user.
  // Segment's analytics.user().id() reflects that persisted identity
  // directly, so check it in addition to the ref.
  //
  // analytics.user() isn't populated until the async SDK finishes loading —
  // on the bootstrap stub it can read back no ID and we'd never re-check.
  // Defer to analytics.ready(), and guard with a cancelled flag so a stale
  // callback (e.g. the user signs back in before ready() fires) can't reset
  // the identity we just set for them.
  useEffect(() => {
    if (typeof window === "undefined" || !window.analytics) return;
    if (loading || user) return

    let cancelled = false;

    window.analytics.ready(() => {
      if (cancelled) return;

      const persistedUserId = window.analytics.user?.()?.id?.();
      if (identifiedUserIdRef.current === null && !persistedUserId) return;

      identifiedUserIdRef.current = null;
      window.analytics.reset();
    });

    return () => {
      cancelled = true;
    };
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
