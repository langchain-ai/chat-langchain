/**
 * Embeddings module for managing different embedding providers.
 *
 * Supports:
 * - ollama/nomic-embed-text: Local Ollama embeddings with 2K context window (default)
 * - openai/*: OpenAI embeddings
 * - weaviate/*: Legacy Weaviate built-in vectorizer (deprecated)
 */

import { Embeddings } from '@langchain/core/embeddings'
import { OpenAIEmbeddings } from '@langchain/openai'
import { OllamaEmbeddings } from '@langchain/ollama'
import { OLLAMA_BASE_URL } from './constants.js'

/**
 * Get embeddings model based on provider and model name.
 *
 * @param model - Model specification in format "provider/model-name"
 * @param baseUrl - Base URL for Ollama (optional, defaults to OLLAMA_BASE_URL)
 * @returns Embeddings instance or null for Weaviate built-in vectorizer
 *
 * @example
 * ```typescript
 * // Ollama embeddings
 * const embeddings = getEmbeddingsModel("ollama/nomic-embed-text");
 *
 * // OpenAI embeddings
 * const embeddings = getEmbeddingsModel("openai/text-embedding-3-small");
 *
 * // Weaviate built-in (returns null)
 * const embeddings = getEmbeddingsModel("weaviate/vectorizer");
 * ```
 */
export function getEmbeddingsModel(
  model?: string,
  baseUrl: string = OLLAMA_BASE_URL,
): Embeddings | null {
  const ollamaApiKey = process.env.OLLAMA_API_KEY || ''

  const modelSpec =
    model || process.env.EMBEDDING_MODEL || 'ollama/nomic-embed-text'

  const [provider, modelName] = modelSpec.split('/', 2)

  switch (provider.toLowerCase()) {
    case 'ollama':
      // Ollama embeddings with nomic-embed-text (2K context, 768 dimensions)
      return new OllamaEmbeddings({
        model: modelName,
        baseUrl,
        headers: {
          'X-API-Key': ollamaApiKey,
        },
      })

    case 'openai':
      return new OpenAIEmbeddings({
        model: modelName,
        // Chunk size for batching
        batchSize: 200,
      })

    case 'weaviate':
      // Weaviate's built-in vectorizer handles embeddings internally
      // Return null to signal that no external embedding is needed
      return null

    default:
      throw new Error(`Unsupported embedding provider: ${provider}`)
  }
}
