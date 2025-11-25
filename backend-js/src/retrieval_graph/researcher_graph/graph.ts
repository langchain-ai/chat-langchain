/**
 * Researcher graph used in the conversational retrieval system as a subgraph.
 *
 * This module defines the core structure and functionality of the researcher graph,
 * which is responsible for generating search queries and retrieving relevant documents.
 */

import { RunnableConfig } from '@langchain/core/runnables'
import { StateGraph, START, END, Send } from '@langchain/langgraph'
import { z } from 'zod'
import {
  ResearcherStateAnnotation,
  QueryStateAnnotation,
  QueryState,
} from './state.js'
import { getAgentConfiguration } from '../configuration.js'
import { getGenerateQueriesSystemPrompt } from '../prompts.js'
import { loadChatModel } from '../../utils.js'
import { makeRetriever } from '../../retrieval.js'
/**
 * Schema for query generation response
 */
const GenerateQueriesSchema = z.object({
  queries: z.array(z.string()).describe('List of search queries to execute'),
})

/**
 * Generate search queries based on the question (a step in the research plan).
 *
 * This function uses a language model to generate diverse search queries to help answer the question.
 *
 * @param state - The current state of the researcher, including the user's question
 * @param config - Configuration with the model used to generate queries
 * @returns Updated state with generated queries
 */
async function generateQueries(
  state: typeof ResearcherStateAnnotation.State,
  config?: RunnableConfig,
): Promise<Partial<typeof ResearcherStateAnnotation.State>> {
  const configuration = getAgentConfiguration(config)
  const systemPrompt = await getGenerateQueriesSystemPrompt()

  // Determine if we should use function calling method
  const useFunctionCalling = configuration.queryModel.includes('openai')

  const model = loadChatModel(configuration.queryModel)
  const structuredModel = model.withStructuredOutput(GenerateQueriesSchema, {
    method: useFunctionCalling ? 'functionCalling' : 'json_schema',
  })

  const messages = [
    { role: 'system' as const, content: systemPrompt },
    { role: 'human' as const, content: state.question },
  ]

  const response = await structuredModel.invoke(messages, {
    ...config,
    tags: ['langsmith:nostream'],
  })

  return {
    queries: response.queries,
  }
}

/**
 * Retrieve documents based on a given query.
 *
 * This function uses a retriever to fetch relevant documents for a given query.
 *
 * @param state - The current state containing the query string
 * @param config - Configuration with the retriever used to fetch documents
 * @returns Updated state with retrieved documents
 */
async function retrieveDocuments(
  state: typeof QueryStateAnnotation.State,
  config?: RunnableConfig,
): Promise<Partial<typeof QueryStateAnnotation.State>> {
  const retriever = await makeRetriever(config)

  const documents = await retriever.invoke(state.query, config)

  return {
    documents,
    queryIndex: state.queryIndex,
  }
}

/**
 * Create parallel retrieval tasks for each generated query.
 *
 * This function prepares parallel document retrieval tasks for each query in the researcher's state.
 *
 * @param state - The current state of the researcher, including the generated queries
 * @returns List of Send objects, each representing a document retrieval task
 */
function retrieveInParallel(
  state: typeof ResearcherStateAnnotation.State,
): Send<string, QueryState>[] {
  return state.queries.map(
    (query, index) =>
      new Send<string, QueryState>('retrieve_documents', {
        query,
        queryIndex: index,
        documents: [],
      } as QueryState),
  )
}

/**
 * Build and compile the researcher graph
 */
const builder = new StateGraph(ResearcherStateAnnotation)
  .addNode('generate_queries', generateQueries)
  .addNode('retrieve_documents', retrieveDocuments)
  .addEdge(START, 'generate_queries')
  .addConditionalEdges('generate_queries', retrieveInParallel, [
    'retrieve_documents',
  ])
  .addEdge('retrieve_documents', END)

export const graph = builder.compile()
graph.name = 'ResearcherGraph'
