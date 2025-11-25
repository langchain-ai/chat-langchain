/**
 * Agent configuration for the retrieval graph.
 *
 * This module defines the configurable parameters specific to the agent,
 * including model selections and prompt templates.
 */

import { RunnableConfig } from '@langchain/core/runnables'
import { z } from 'zod'
import { BaseConfigurationSchema } from '../configuration.js'

/**
 * Note: Prompts will be loaded from LangSmith at runtime.
 * These are placeholder defaults.
 */
const DEFAULT_PROMPTS = {
  ROUTER_SYSTEM_PROMPT:
    'You are a helpful assistant that routes user questions.',
  MORE_INFO_SYSTEM_PROMPT:
    'You are a helpful assistant that asks for clarification.',
  GENERAL_SYSTEM_PROMPT: 'You are a helpful assistant.',
  RESEARCH_PLAN_SYSTEM_PROMPT:
    "You are a research planner. Create a step-by-step research plan to answer the user's question about LangChain.",
  GENERATE_QUERIES_SYSTEM_PROMPT:
    'Generate diverse search queries to help answer the research question.',
  RESPONSE_SYSTEM_PROMPT:
    "You are an expert on LangChain. Answer the user's question based on the provided context.\n\nContext:\n{context}",
}

/**
 * AgentConfiguration extends BaseConfiguration with agent-specific settings
 */
export const AgentConfigurationSchema = BaseConfigurationSchema.extend({
  /**
   * The language model used for processing and refining queries.
   * Should be in the form: provider/model-name.
   * @default "groq/openai/gpt-oss-20b"
   */
  queryModel: z
    .string()
    .default('groq/openai/gpt-oss-20b')
    .describe('The language model used for query processing'),

  /**
   * The language model used for generating responses.
   * Should be in the form: provider/model-name.
   * @default "groq/openai/gpt-oss-20b"
   */
  responseModel: z
    .string()
    .default('groq/openai/gpt-oss-20b')
    .describe('The language model used for generating responses'),

  /**
   * System prompts for different stages of the agent
   */
  routerSystemPrompt: z
    .string()
    .default(DEFAULT_PROMPTS.ROUTER_SYSTEM_PROMPT)
    .describe('System prompt for routing user questions'),

  moreInfoSystemPrompt: z
    .string()
    .default(DEFAULT_PROMPTS.MORE_INFO_SYSTEM_PROMPT)
    .describe('System prompt for asking for more information'),

  generalSystemPrompt: z
    .string()
    .default(DEFAULT_PROMPTS.GENERAL_SYSTEM_PROMPT)
    .describe('System prompt for general questions'),

  researchPlanSystemPrompt: z
    .string()
    .default(DEFAULT_PROMPTS.RESEARCH_PLAN_SYSTEM_PROMPT)
    .describe('System prompt for generating research plans'),

  generateQueriesSystemPrompt: z
    .string()
    .default(DEFAULT_PROMPTS.GENERATE_QUERIES_SYSTEM_PROMPT)
    .describe('System prompt for generating search queries'),

  responseSystemPrompt: z
    .string()
    .default(DEFAULT_PROMPTS.RESPONSE_SYSTEM_PROMPT)
    .describe('System prompt for generating final responses'),
})

export type AgentConfiguration = z.infer<typeof AgentConfigurationSchema>

/**
 * Extract agent configuration from RunnableConfig
 *
 * Reads from config.configurable to match Python's behavior
 * This is the standard approach used by LangSmith
 */
export function getAgentConfiguration(
  config?: RunnableConfig,
): AgentConfiguration {
  // Use configurable (matches Python's from_runnable_config)
  const configurable = config?.configurable || {}

  // Convert snake_case to camelCase for all fields
  const camelCased: Record<string, any> = {}

  for (const [key, value] of Object.entries(configurable)) {
    // Convert snake_case to camelCase
    const camelKey = key.replace(/_([a-z])/g, (_, letter) =>
      letter.toUpperCase(),
    )
    camelCased[camelKey] = value
  }

  // Parse and validate with defaults
  return AgentConfigurationSchema.parse(camelCased)
}

/**
 * Load prompts from LangSmith (to be implemented with actual LangSmith client)
 * For now, returns empty object to use schema defaults
 */
export async function loadPromptsFromLangSmith(): Promise<
  Partial<AgentConfiguration>
> {
  // TODO: Implement actual LangSmith prompt loading
  // This would use langsmith SDK to fetch prompts
  // For now, return empty object and let schema defaults apply
  return {}
}
