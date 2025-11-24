/**
 * Retrieval module for creating and managing retrievers.
 *
 * This module provides factory functions for creating retrievers
 * based on the current configuration.
 */

import { VectorStoreRetriever } from '@langchain/core/vectorstores'
import { RunnableConfig } from '@langchain/core/runnables'
import { WeaviateStore } from '@langchain/weaviate'
import { getWeaviateClient } from './utils.js'
import { getEmbeddingsModel } from './embeddings.js'
import { getBaseConfiguration } from './configuration.js'
import {
  OLLAMA_BASE_URL,
  WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
} from './constants.js'

const WEAVIATE_URL = process.env.WEAVIATE_URL
const WEAVIATE_GRPC_URL = process.env.WEAVIATE_GRPC_URL
const WEAVIATE_API_KEY = process.env.WEAVIATE_API_KEY

/**
 * Create a Weaviate retriever with the specified configuration.
 *
 * @param embeddingModel - The embedding model specification
 * @param searchKwargs - Additional search parameters
 * @param baseUrl - Base URL for Ollama embeddings
 * @returns A VectorStoreRetriever configured for Weaviate
 */
export async function makeWeaviateRetriever(
  embeddingModel: string,
  searchKwargs: Record<string, any> = {},
  baseUrl: string = OLLAMA_BASE_URL,
): Promise<VectorStoreRetriever> {
  const client = await getWeaviateClient(
    WEAVIATE_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_API_KEY,
  )

  const embeddings = getEmbeddingsModel(embeddingModel, baseUrl)

  if (!embeddings) {
    throw new Error(
      'Weaviate built-in vectorizer not yet supported in TypeScript version',
    )
  }

  const store = new WeaviateStore(embeddings, {
    client,
    indexName: WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
    textKey: 'text',
    metadataKeys: ['source', 'title'],
  })

  // Merge default search kwargs with provided ones
  const finalSearchKwargs = {
    k: 6,
    ...searchKwargs,
  }

  return store.asRetriever(finalSearchKwargs)
}

/**
 * Create a retriever based on the provided configuration.
 *
 * This is the main factory function that routes to the appropriate
 * retriever implementation based on the configuration.
 *
 * @param config - RunnableConfig containing retriever configuration
 * @param baseUrl - Base URL for Ollama (optional)
 * @returns A configured retriever
 *
 * @example
 * ```typescript
 * const retriever = await makeRetriever(config);
 * const docs = await retriever.invoke("What is LangChain?");
 * ```
 */
export async function makeRetriever(
  config?: RunnableConfig,
  baseUrl: string = OLLAMA_BASE_URL,
): Promise<VectorStoreRetriever> {
  const configuration = getBaseConfiguration(config)

  switch (configuration.retrieverProvider) {
    case 'weaviate':
      return makeWeaviateRetriever(
        configuration.embeddingModel,
        configuration.searchKwargs,
        baseUrl,
      )

    default:
      throw new Error(
        `Unrecognized retriever_provider in configuration. ` +
          `Expected: weaviate, Got: ${configuration.retrieverProvider}`,
      )
  }
}

/**
 * Helper function to clean up Weaviate client connection.
 * Call this when done with retrieval operations.
 */
export async function closeWeaviateClient(
  retriever: VectorStoreRetriever,
): Promise<void> {
  // The retriever's vectorStore should have access to the client
  const store = retriever.vectorStore as WeaviateStore
  // Access client via type assertion since it's private but we need to close it
  if (store && (store as any).client) {
    await (store as any).client.close()
  }
}
