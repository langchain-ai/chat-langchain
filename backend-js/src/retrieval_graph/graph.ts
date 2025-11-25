/**
 * Main entrypoint for the conversational retrieval graph.
 *
 * This module defines the core structure and functionality of the conversational
 * retrieval graph. It includes the main graph definition, state management,
 * and key functions for processing & routing user queries, generating research plans to answer user questions,
 * conducting research, and formulating responses.
 */

import { RunnableConfig } from '@langchain/core/runnables'
import { StateGraph, START, END } from '@langchain/langgraph'
import { z } from 'zod'
import { AgentStateAnnotation, InputStateAnnotation } from './state.js'
import {
  AgentConfigurationSchema,
  getAgentConfiguration,
} from './configuration.js'
import {
  getResearchPlanSystemPrompt,
  getResponseSystemPrompt,
} from './prompts.js'
import { loadChatModel, formatDocs } from '../utils.js'
import { graph as researcherGraph } from './researcher_graph/graph.js'

/**
 * Schema for research plan
 */
const ResearchPlanSchema = z.object({
  steps: z.array(z.string()).describe('List of research steps to complete'),
})

/**
 * Create a step-by-step research plan for answering a LangChain-related query.
 *
 * @param state - The current state of the agent, including conversation history
 * @param config - Configuration with the model used to generate the plan
 * @returns Updated state with research steps
 */
async function createResearchPlan(
  state: typeof AgentStateAnnotation.State,
  config?: RunnableConfig,
): Promise<
  | Partial<typeof AgentStateAnnotation.State>
  | { steps: string[]; documents: 'delete'; query: string }
> {
  const configuration = getAgentConfiguration(config)
  const systemPrompt = await getResearchPlanSystemPrompt()

  // Determine if we should use function calling method
  const useFunctionCalling = configuration.queryModel.includes('openai')

  const model = loadChatModel(configuration.queryModel)
  const structuredModel = model.withStructuredOutput(ResearchPlanSchema, {
    method: useFunctionCalling ? 'functionCalling' : 'json_schema',
  })

  const messages = [
    { role: 'system' as const, content: systemPrompt },
    ...state.messages,
  ]

  const response = await structuredModel.invoke(messages, {
    ...config,
    tags: ['langsmith:nostream'],
  })

  // Get the query from the last message
  const lastMessage = state.messages[state.messages.length - 1]
  const query = 'content' in lastMessage ? String(lastMessage.content) : ''

  return {
    steps: response.steps,
    documents: 'delete' as 'delete',
    query,
  }
}

/**
 * Execute the first step of the research plan.
 *
 * This function takes the first step from the research plan and uses it to conduct research.
 *
 * @param state - The current state of the agent, including the research plan steps
 * @param config - Configuration for the research
 * @returns Updated state with retrieved documents and remaining steps
 */
async function conductResearch(
  state: typeof AgentStateAnnotation.State,
  config?: RunnableConfig,
): Promise<Partial<typeof AgentStateAnnotation.State>> {
  if (!state.steps || state.steps.length === 0) {
    return { steps: [] }
  }

  const result = await researcherGraph.invoke(
    { question: state.steps[0] },
    config,
  )

  return {
    documents: result.documents,
    steps: state.steps.slice(1),
  }
}

/**
 * Determine if the research process is complete or if more research is needed.
 *
 * This function checks if there are any remaining steps in the research plan.
 *
 * @param state - The current state of the agent, including the remaining research steps
 * @returns The next step to take based on whether research is complete
 */
function checkFinished(
  state: typeof AgentStateAnnotation.State,
): 'respond' | 'conduct_research' {
  if (state.steps && state.steps.length > 0) {
    return 'conduct_research'
  }
  return 'respond'
}

/**
 * Generate a final response to the user's query based on the conducted research.
 *
 * This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.
 *
 * @param state - The current state of the agent, including retrieved documents and conversation history
 * @param config - Configuration with the model used to respond
 * @returns Updated state with the generated response
 */
async function respond(
  state: typeof AgentStateAnnotation.State,
  config?: RunnableConfig,
): Promise<Partial<typeof AgentStateAnnotation.State>> {
  const configuration = getAgentConfiguration(config)
  const model = loadChatModel(configuration.responseModel)

  // TODO: add a re-ranker here
  const topK = 3
  const documents = state.documents || []
  const context = formatDocs(documents.slice(0, topK))

  const systemPromptTemplate = await getResponseSystemPrompt()
  const systemPrompt = systemPromptTemplate.replace('{context}', context)

  const messages = [
    { role: 'system' as const, content: systemPrompt },
    ...state.messages,
  ]

  const response = await model.invoke(messages, config)

  // Extract answer content - match Python's response.content behavior
  const answerContent =
    typeof response.content === 'string'
      ? response.content
      : JSON.stringify(response.content)

  // Ensure answer is a non-empty string
  const finalAnswer = answerContent || 'No answer generated'

  return {
    messages: [response],
    answer: finalAnswer,
  }
}

const builder = new StateGraph(AgentStateAnnotation, {
  input: InputStateAnnotation,
  context: AgentConfigurationSchema,
})
  .addNode('create_research_plan', createResearchPlan)
  .addNode('conduct_research', conductResearch)
  .addNode('respond', respond)
  .addEdge(START, 'create_research_plan')
  .addEdge('create_research_plan', 'conduct_research')
  .addConditionalEdges('conduct_research', checkFinished)
  .addEdge('respond', END)

export const graph = builder.compile()
graph.name = 'RetrievalGraph'
