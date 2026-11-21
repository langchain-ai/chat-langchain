// The sign-in prompt opens on its own once the auth check settles. By then a
// guest may already be typing in the composer (the guest session usually
// resolves first). Opening the dialog mid-typing steals focus, so an Enter
// pressed at that moment never reaches the composer and the first message is
// silently dropped (sometimes with the draft). Skip the automatic prompt once
// the user has started writing; the header's Sign in button still opens it.
export function shouldAutoOpenAuthModal(state: {
  enabled: boolean
  loading: boolean
  signedIn: boolean
  dismissed: boolean
  userStartedTyping: boolean
}): boolean {
  if (!state.enabled || state.loading || state.signedIn) return false
  if (state.dismissed) return false
  return !state.userStartedTyping
}

export function isComposerTypingTarget(target: EventTarget | null): boolean {
  if (!target || typeof target !== "object") return false
  const el = target as { tagName?: string; isContentEditable?: boolean; closest?: (s: string) => unknown }
  if (el.closest?.("[role=dialog]")) return false
  return el.tagName === "TEXTAREA" || el.isContentEditable === true
}
