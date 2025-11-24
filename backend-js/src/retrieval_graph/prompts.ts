/**
 * Prompt templates for the retrieval graph.
 *
 * This module loads prompts from LangSmith for consistent prompt management.
 * Falls back to default prompts if LangSmith is not available.
 */

import { Client } from 'langsmith'

// Initialize LangSmith client
const client = new Client({
  apiKey: process.env.LANGCHAIN_PROMPT_API_KEY || process.env.LANGCHAIN_API_KEY,
  apiUrl: process.env.LANGCHAIN_PROMPT_API_URL,
})

/**
 * Default prompts (fallback if LangSmith is unavailable)
 */
const DEFAULT_PROMPTS = {
  ROUTER_SYSTEM_PROMPT: `You are an expert at routing user questions to the appropriate handler.
Given a user question, determine if it needs research or can be answered directly.`,

  MORE_INFO_SYSTEM_PROMPT: `You are a helpful assistant that asks for clarification when user questions are unclear or ambiguous.
Ask specific questions to better understand what the user needs.`,

  GENERAL_SYSTEM_PROMPT: `You are a helpful AI assistant with expertise in LangChain and related technologies.
Provide clear, accurate, and helpful responses to user questions.`,

  RESEARCH_PLAN_SYSTEM_PROMPT: `You are a research planner for LangChain-related queries.
Break down the user's question into a step-by-step research plan.
Each step should be a specific, focused research question that can be answered through document retrieval.
Generate 2-4 research steps.`,

  GENERATE_QUERIES_SYSTEM_PROMPT: `You are an expert at generating diverse search queries.
Given a research question about LangChain, generate 3-5 diverse search queries that would help find relevant information.
The queries should approach the topic from different angles to maximize coverage.`,

  RESPONSE_SYSTEM_PROMPT: `You are an expert on LangChain and related technologies.
Answer the user's question based on the provided context from the documentation.
If the context doesn't contain enough information, acknowledge this.
Be specific and cite the documentation when appropriate.

Context:
{context}`,
}

/**
 * Cached prompts to avoid repeated API calls
 */
let cachedPrompts: Record<string, string> | null = null

/**
 * Load a single prompt from LangSmith
 */
async function loadPromptFromLangSmith(promptName: string): Promise<string> {
  try {
    const prompt = await client._pullPrompt(promptName)
    // Extract the template from the prompt
    if (prompt && prompt.messages && prompt.messages.length > 0) {
      const firstMessage = prompt.messages[0]
      if (firstMessage && 'prompt' in firstMessage && firstMessage.prompt) {
        return firstMessage.prompt.template || ''
      }
    }
    return ''
  } catch (error) {
    console.warn(`Failed to load prompt ${promptName} from LangSmith:`, error)
    return ''
  }
}

/**
 * Load all prompts from LangSmith or use defaults
 */
async function loadAllPrompts(): Promise<Record<string, string>> {
  if (cachedPrompts) {
    return cachedPrompts
  }

  const prompts: Record<string, string> = {}

  try {
    // Try to load prompts from LangSmith
    const [router, moreInfo, general, researchPlan, generateQueries, response] =
      await Promise.allSettled([
        loadPromptFromLangSmith('langchain-ai/chat-langchain-router-prompt'),
        loadPromptFromLangSmith('langchain-ai/chat-langchain-more-info-prompt'),
        loadPromptFromLangSmith('langchain-ai/chat-langchain-general-prompt'),
        loadPromptFromLangSmith(
          'langchain-ai/chat-langchain-research-plan-prompt',
        ),
        loadPromptFromLangSmith(
          'langchain-ai/chat-langchain-generate-queries-prompt',
        ),
        loadPromptFromLangSmith('langchain-ai/chat-langchain-response-prompt'),
      ])

    // Use LangSmith prompts if available, otherwise fall back to defaults
    prompts.ROUTER_SYSTEM_PROMPT =
      (router.status === 'fulfilled' && router.value) ||
      DEFAULT_PROMPTS.ROUTER_SYSTEM_PROMPT
    prompts.MORE_INFO_SYSTEM_PROMPT =
      (moreInfo.status === 'fulfilled' && moreInfo.value) ||
      DEFAULT_PROMPTS.MORE_INFO_SYSTEM_PROMPT
    prompts.GENERAL_SYSTEM_PROMPT =
      (general.status === 'fulfilled' && general.value) ||
      DEFAULT_PROMPTS.GENERAL_SYSTEM_PROMPT
    prompts.RESEARCH_PLAN_SYSTEM_PROMPT =
      (researchPlan.status === 'fulfilled' && researchPlan.value) ||
      DEFAULT_PROMPTS.RESEARCH_PLAN_SYSTEM_PROMPT
    prompts.GENERATE_QUERIES_SYSTEM_PROMPT =
      (generateQueries.status === 'fulfilled' && generateQueries.value) ||
      DEFAULT_PROMPTS.GENERATE_QUERIES_SYSTEM_PROMPT
    prompts.RESPONSE_SYSTEM_PROMPT =
      (response.status === 'fulfilled' && response.value) ||
      DEFAULT_PROMPTS.RESPONSE_SYSTEM_PROMPT
  } catch (error) {
    console.warn(
      'Failed to load prompts from LangSmith, using defaults:',
      error,
    )
    Object.assign(prompts, DEFAULT_PROMPTS)
  }

  cachedPrompts = prompts
  return prompts
}

// Load prompts on module initialization
let promptsPromise: Promise<Record<string, string>> | null = null

/**
 * Get all prompts (loads from LangSmith on first call, then caches)
 */
export async function getPrompts(): Promise<Record<string, string>> {
  if (!promptsPromise) {
    promptsPromise = loadAllPrompts()
  }
  return promptsPromise
}

// Export individual prompt getters for convenience
export async function getRouterSystemPrompt(): Promise<string> {
  const prompts = await getPrompts()
  return prompts.ROUTER_SYSTEM_PROMPT
}

export async function getMoreInfoSystemPrompt(): Promise<string> {
  const prompts = await getPrompts()
  return prompts.MORE_INFO_SYSTEM_PROMPT
}

export async function getGeneralSystemPrompt(): Promise<string> {
  const prompts = await getPrompts()
  return prompts.GENERAL_SYSTEM_PROMPT
}

export async function getResearchPlanSystemPrompt(): Promise<string> {
  const prompts = await getPrompts()
  return prompts.RESEARCH_PLAN_SYSTEM_PROMPT
}

export async function getGenerateQueriesSystemPrompt(): Promise<string> {
  const prompts = await getPrompts()
  return prompts.GENERATE_QUERIES_SYSTEM_PROMPT
}

export async function getResponseSystemPrompt(): Promise<string> {
  const prompts = await getPrompts()
  return prompts.RESPONSE_SYSTEM_PROMPT
}
