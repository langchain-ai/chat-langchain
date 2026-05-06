/**
 * Deployment Configuration
 *
 * This file manages configuration differences between internal and external deployments.
 *
 * - Internal: Full access for team use (all models, all agents, Google OAuth protected)
 * - External: Locked down for customer-facing use (limited models, docs_agent only, no auth)
 */

// =============================================================================
// Config Storage
// =============================================================================

/** Bump version to force reset of saved user configs */
export const CONFIG_STORAGE = {
  key: "agent-config",
  versionKey: "agent-config-version",
  version: "0.5",
} as const

// =============================================================================
// Model Registry
// =============================================================================

interface ModelConfig {
  id: string // e.g., "xai:grok-4-1-fast-non-reasoning"
  name: string // Display name
  provider: "xai" | "anthropic" | "openai" | "openrouter" | "google" | "baseten"
  description?: string
}

/**
 * All available models - single source of truth
 * Keys are short names, values contain full configuration
 */
export const MODELS = {
  "grok-4.1-fast": {
    id: "xai:grok-4-1-fast-non-reasoning",
    name: "Grok 4.1 Fast",
    provider: "xai",
    description: "Fast, optimized for tool calling",
  },
  "claude-sonnet-4.6": {
    id: "anthropic:claude-sonnet-4-6",
    name: "Claude Sonnet 4.6",
    provider: "anthropic",
    description: "Latest, most capable Sonnet",
  },
  "gpt-5.4-nano": {
    id: "openai:gpt-5.4-nano",
    name: "GPT-5.4 Nano",
    provider: "openai",
    description: "Cheapest GPT-5.4-class model for simple high-volume tasks",
  },
  "gpt-5.4-mini": {
    id: "openai:gpt-5.4-mini",
    name: "GPT-5.4 Mini",
    provider: "openai",
    description: "Strongest mini model for coding, computer use, and subagents",
  },
  "gemini-2.5-flash": {
    id: "google_genai:gemini-2.5-flash",
    name: "Gemini 2.5 Flash",
    provider: "google",
    description: "Fast and capable Google model",
  },
  "gemini-3.1-flash-lite": {
    id: "google_genai:gemini-3.1-flash-lite-preview",
    name: "Gemini 3.1 Flash Lite",
    provider: "google",
    description: "Fastest, most cost-effective Gemini",
  },
  "gemini-3-pro": {
    id: "google_genai:gemini-3-pro-preview",
    name: "Gemini 3 Pro",
    provider: "google",
    description: "Most capable Gemini model",
  },
  "glm-5": {
    id: "baseten:zai-org/GLM-5",
    name: "GLM 5",
    provider: "baseten",
    description: "Z.ai GLM 5 served via Baseten",
  },
} as const satisfies Record<string, ModelConfig>

export type ModelKey = keyof typeof MODELS
export type ModelOption = (typeof MODELS)[ModelKey]["id"]

// =============================================================================
// Agent Registry
// =============================================================================

interface AgentConfig {
  id: string
  name: string
  shortName: string
  description?: string
}

export const AGENTS = {
  docs: {
    id: "docs_agent",
    name: "Docs Agent",
    shortName: "Docs Agent",
    description: "Documentation search and Q&A",
  },
} as const satisfies Record<string, AgentConfig>

export type AgentKey = keyof typeof AGENTS
export type AgentType = (typeof AGENTS)[AgentKey]["id"]

// =============================================================================
// Deployment Configuration
// =============================================================================

export type DeploymentEnv = "internal" | "external"

interface DeploymentConfig {
  models: ModelKey[]
  agents: AgentKey[]
  defaultModel: ModelKey
  defaultAgent: AgentKey
  requiresAuth: boolean
}

const DEPLOYMENTS: Record<DeploymentEnv, DeploymentConfig> = {
  external: {
    models: ["gpt-5.4-mini", "gemini-3.1-flash-lite", "glm-5"],
    agents: ["docs"],
    defaultModel: "gemini-3.1-flash-lite",
    defaultAgent: "docs",
    requiresAuth: false,
  },
  internal: {
    models: ["gpt-5.4-mini", "gemini-3.1-flash-lite", "glm-5"],
    agents: ["docs"],
    defaultModel: "gemini-3.1-flash-lite",
    defaultAgent: "docs",
    requiresAuth: false,
  },
}

// =============================================================================
// Core Functions
// =============================================================================

export function getDeploymentEnv(): DeploymentEnv {
  const env = process.env.NEXT_PUBLIC_DEPLOYMENT_ENV
  return env === "internal" ? "internal" : "external"
}

export function getDeploymentConfig(): DeploymentConfig {
  return DEPLOYMENTS[getDeploymentEnv()]
}

// =============================================================================
// Model Functions
// =============================================================================

export function getAllowedModels(): ModelOption[] {
  return getDeploymentConfig().models.map((key) => MODELS[key].id)
}

export function getDefaultModel(): ModelOption {
  return MODELS[getDeploymentConfig().defaultModel].id
}

export function isModelAllowed(modelId: ModelOption): boolean {
  return getAllowedModels().includes(modelId)
}

export function getModelDisplayName(modelId: ModelOption): string {
  const model = Object.values(MODELS).find((m) => m.id === modelId)
  return model?.name ?? modelId
}

export function getModelProvider(modelId: ModelOption): string {
  const model = Object.values(MODELS).find((m) => m.id === modelId)
  return model?.provider ?? "openai"
}

// =============================================================================
// Agent Functions
// =============================================================================

export function getAllowedAgents(): AgentType[] {
  return getDeploymentConfig().agents.map((key) => AGENTS[key].id)
}

export function getDefaultAgent(): AgentType {
  return AGENTS[getDeploymentConfig().defaultAgent].id
}

export function isAgentAllowed(agentId: AgentType): boolean {
  return getAllowedAgents().includes(agentId)
}

export function getAgentDisplayName(agentId: AgentType): string {
  const agent = Object.values(AGENTS).find((a) => a.id === agentId)
  return agent?.name ?? agentId
}

export function getAgentShortDisplayName(agentId: AgentType): string {
  const agent = Object.values(AGENTS).find((a) => a.id === agentId)
  return agent?.shortName ?? agentId
}

// =============================================================================
// Auth Functions
// =============================================================================

export function isAuthRequired(): boolean {
  return getDeploymentConfig().requiresAuth
}
