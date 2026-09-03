"use client";

import { createContext, useCallback, useContext, useEffect, useRef } from "react";
import { useAuth } from "@/lib/auth";

// Declare analytics global type
declare global {
  interface Window {
    analytics: any;
  }
}

/**
 * Exposes a live accessor for Segment's anonymous ID rather than a cached
 * value. The ID can be regenerated asynchronously by Segment (e.g. after
 * reset() on logout), so callers should read it at the moment they need it
 * instead of relying on a React-state snapshot that could go stale.
 */
interface AnalyticsContextValue {
  getAnonymousId: () => string | null;
}

const AnalyticsContext = createContext<AnalyticsContextValue>({
  getAnonymousId: () => null,
});

export function useAnalyticsContext(): AnalyticsContextValue {
  return useContext(AnalyticsContext);
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
  // Tracks whether Segment's SDK has finished loading. Set once inside
  // analytics.ready() below; never cleared, since readiness doesn't regress.
  const isReadyRef = useRef(false);

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

  // Track when Segment's SDK has finished loading, independent of auth
  // state, so getAnonymousId() below knows when it's safe to read from it.
  useEffect(() => {
    if (typeof window === "undefined" || !window.analytics) return;

    let cancelled = false;

    window.analytics.ready(() => {
      if (!cancelled) isReadyRef.current = true;
    });

    return () => {
      cancelled = true;
    };
  }, []);

  // Live accessor rather than cached state: Segment can regenerate the
  // anonymous ID asynchronously (e.g. after reset() on logout), so callers
  // should read the current value at the moment they need it rather than a
  // snapshot that could go stale.
  const getAnonymousId = useCallback((): string | null => {
    if (typeof window === "undefined" || !window.analytics || !isReadyRef.current) {
      return null;
    }

    try {
      return window.analytics.user?.()?.anonymousId?.() || null;
    } catch {
      return null;
    }
  }, []);

  return (
    <AnalyticsContext.Provider value={{ getAnonymousId }}>
      {children}
    </AnalyticsContext.Provider>
  );
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
