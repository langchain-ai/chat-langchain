import type { ClientProfile } from "@/lib/hooks/threads"

// Constants
const DEFAULT_AVATAR_COLOR = "#6366f1" // Indigo-500
const FALLBACK_CLIENT: ClientProfile = {
  id: "local-client",
  label: "Local Session",
  avatarColor: DEFAULT_AVATAR_COLOR,
}

/**
 * Derives a human-readable label for a client profile.
 * If no label is provided, generates one from the client ID (e.g., "Client A1B2").
 */
function deriveLabel(client: Pick<ClientProfile, 'id' | 'label'>): string {
  if (client.label?.trim()) {
    return client.label.trim()
  }

  const suffix = client.id.slice(-4).toUpperCase()
  return `Client ${suffix}`
}

/**
 * Generates a unique client ID using crypto.randomUUID() if available,
 * otherwise falls back to a random string.
 */
function generateClientId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }

  return `client-${Math.random().toString(36).slice(2, 10)}`
}

/**
 * Resolves a partial client profile into a complete ClientProfile object.
 * Fills in missing fields with fallback values and derives a label if needed.
 */
export function resolveClientProfile(input?: Partial<ClientProfile>): ClientProfile {
  if (!input) {
    return { ...FALLBACK_CLIENT }
  }

  const id = input.id?.trim() || FALLBACK_CLIENT.id
  const label = deriveLabel({ id, label: input.label })
  const avatarColor = input.avatarColor || DEFAULT_AVATAR_COLOR

  return {
    id,
    label,
    avatarColor,
  }
}

/**
 * Creates a new client profile with optional overrides.
 * Generates a random ID if not provided and resolves all fields.
 */
export function createClientProfile(overrides?: Partial<ClientProfile>): ClientProfile {
  const base: Partial<ClientProfile> = {
    id: overrides?.id ?? generateClientId(),
    label: overrides?.label,
    avatarColor: overrides?.avatarColor,
  }

  return resolveClientProfile(base)
}
