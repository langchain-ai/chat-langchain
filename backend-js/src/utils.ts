/**
 * Shared utility functions used in the project.
 *
 * Functions:
 *   - getWeaviateClient: Create a Weaviate client connection
 *   - formatDocs: Convert documents to an xml-formatted string
 *   - loadChatModel: Load a chat model from a model name
 *   - reduceDocs: Document reducer for state management
 */

import weaviate, { WeaviateClient } from 'weaviate-client'
import { Document } from '@langchain/core/documents'
import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatOpenAI } from '@langchain/openai'
import { ChatGroq } from '@langchain/groq'
import { ChatOllama } from '@langchain/ollama'
import { v4 as uuidv4 } from 'uuid'

/**
 * Create and connect to a Weaviate client.
 *
 * @param weaviateUrl - The Weaviate HTTP URL. If not provided, reads from WEAVIATE_URL env var.
 * @param weaviateGrpcUrl - The Weaviate gRPC URL. If not provided, uses weaviateUrl.
 * @param weaviateApiKey - The Weaviate API key. If not provided, reads from WEAVIATE_API_KEY env var.
 * @returns A connected Weaviate client
 */
export async function getWeaviateClient(
  weaviateUrl?: string,
  weaviateGrpcUrl?: string,
  weaviateApiKey?: string,
): Promise<WeaviateClient> {
  const url = weaviateUrl || process.env.WEAVIATE_URL || 'weaviate.hanu-nus.com'
  const grpcUrl =
    weaviateGrpcUrl ||
    process.env.WEAVIATE_GRPC_URL ||
    'grpc-weaviate.hanu-nus.com'
  const apiKey = weaviateApiKey || process.env.WEAVIATE_API_KEY || 'admin-key'

  // Extract hostname from URL (remove https:// or http://)
  const httpHost = url.replace(/^https?:\/\//, '')
  const grpcHost = grpcUrl.replace(/^https?:\/\//, '')

  const client = await weaviate.connectToCustom({
    httpHost,
    httpPort: 443,
    httpSecure: true,
    grpcHost,
    grpcPort: 443,
    grpcSecure: true,
    authCredentials: new weaviate.ApiKey(apiKey),
    // Skip init checks to avoid gRPC health check failures with proxied/tunneled connections
    skipInitChecks: true,
    // Increase timeouts for slow/tunneled connections
    timeout: {
      init: 60_000, // 60 seconds for initialization
      query: 60_000, // 60 seconds for queries
      insert: 120_000, // 2 minutes for inserts
    },
  })

  return client
}

/**
 * Format a single document as XML.
 *
 * @param doc - The document to format
 * @returns The formatted document as an XML string
 */
function formatDoc(doc: Document): string {
  const metadata = doc.metadata || {}
  const metaStr = Object.entries(metadata)
    .map(([k, v]) => ` ${k}="${v}"`)
    .join('')

  return `<document${metaStr}>\n${doc.pageContent}\n</document>`
}

/**
 * Format a list of documents as XML.
 *
 * This function takes a list of Document objects and formats them into a single XML string.
 *
 * @param docs - A list of Document objects to format, or null
 * @returns A string containing the formatted documents in XML format
 *
 * @example
 * ```typescript
 * const docs = [
 *   new Document({ pageContent: "Hello" }),
 *   new Document({ pageContent: "World" })
 * ];
 * console.log(formatDocs(docs));
 * // Output:
 * // <documents>
 * // <document>
 * // Hello
 * // </document>
 * // <document>
 * // World
 * // </document>
 * // </documents>
 * ```
 */
export function formatDocs(docs: Document[] | null | undefined): string {
  if (!docs || docs.length === 0) {
    return '<documents></documents>'
  }
  const formatted = docs.map((doc) => formatDoc(doc)).join('\n')
  return `<documents>\n${formatted}\n</documents>`
}

/**
 * Load a chat model from a fully specified name.
 *
 * @param fullySpecifiedName - String in the format 'provider/model'
 * @returns A BaseChatModel instance
 *
 * @example
 * ```typescript
 * // Load Groq model
 * const model = loadChatModel("groq/llama-3.3-70b-versatile");
 * // Load other models
 * const model2 = loadChatModel("openai/gpt-4");
 * ```
 */
export function loadChatModel(fullySpecifiedName: string): BaseChatModel {
  let provider: string
  let model: string

  if (fullySpecifiedName.includes('/')) {
    // Split only on the first '/' to handle formats like "groq/openai/gpt-oss-20b"
    // This matches Python's split("/", maxsplit=1) behavior
    const parts = fullySpecifiedName.split('/')
    provider = parts[0]
    model = parts.slice(1).join('/')
  } else {
    provider = ''
    model = fullySpecifiedName
  }

  const baseConfig = {
    temperature: 0,
  }

  switch (provider.toLowerCase()) {
    case 'groq':
      return new ChatGroq({
        // model: 'llama-3.1-8b-instant',
        // model,
        model: 'llama-3.1-8b-instant', // TODO: change back to model
        ...baseConfig,
      })

    case 'openai':
      return new ChatOpenAI({
        model,
        ...baseConfig,
        streamUsage: true,
      })

    case 'anthropic':
      return new ChatAnthropic({
        model,
        ...baseConfig,
      })

    case 'ollama':
      return new ChatOllama({
        model,
        ...baseConfig,
        baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
      })

    case 'google_genai':
      // Note: Google GenAI might need special handling for system messages
      throw new Error('Google GenAI not yet implemented in TypeScript version')

    default:
      // Default to OpenAI if no provider specified
      return new ChatOpenAI({
        model: fullySpecifiedName,
        ...baseConfig,
        streamUsage: true,
      })
  }
}

/**
 * Reduce and process documents based on the input type.
 *
 * This function handles various input types and converts them into a sequence of Document objects.
 * It also combines existing documents with the new ones based on the document ID.
 *
 * @param existing - The existing docs in the state, if any
 * @param newDocs - The new input to process. Can be a sequence of Documents, objects, strings, or "delete"
 * @returns Combined list of documents
 */
/**
 * Reduce and process documents based on the input type.
 *
 * This function handles various input types and converts them into a sequence of Document objects.
 * It uses dual deduplication: UUID-based (primary) and content-based (secondary).
 * The content-based check handles retrieved documents that have different UUIDs but identical content.
 *
 * @param existing - The existing docs in the state, if any
 * @param newDocs - The new input to process. Can be a sequence of Documents, objects, strings, or "delete"
 * @returns Combined list of documents
 */
export function reduceDocs(
  existing: Document[] | undefined,
  newDocs: Document[] | Record<string, any>[] | string[] | string | 'delete',
): Document[] {
  if (newDocs === 'delete') {
    return []
  }

  const existingList = existing || []

  if (typeof newDocs === 'string') {
    return [
      ...existingList,
      new Document({
        pageContent: newDocs,
        metadata: { uuid: uuidv4() },
      }),
    ]
  }

  if (!Array.isArray(newDocs)) {
    return existingList
  }

  const newList: Document[] = []
  // Primary deduplication: Track UUIDs (matches Python's behavior)
  const existingIds = new Set(
    existingList.map((doc) => doc.metadata?.uuid).filter(Boolean),
  )

  // Secondary deduplication: Track content+source signatures
  // This catches retrieved documents with different UUIDs but identical content
  const existingContentKeys = new Set(
    existingList.map((doc) => {
      const source = doc.metadata?.source || ''
      const content = doc.pageContent.substring(0, 500) // Use first 500 chars as signature
      return `${source}:::${content}`
    }),
  )

  for (const item of newDocs) {
    if (typeof item === 'string') {
      const itemId = uuidv4()
      newList.push(
        new Document({
          pageContent: item,
          metadata: { uuid: itemId },
        }),
      )
      existingIds.add(itemId)
    } else if (item instanceof Document) {
      // Use existing id from Document (from vector DB) if available, otherwise check metadata.uuid, fallback to generating new UUID
      let itemId = item.id || item.metadata?.uuid
      if (!itemId) {
        // Generate new UUID only if neither id nor metadata.uuid exists
        itemId = uuidv4()
      }

      // Primary check: UUID-based deduplication
      if (existingIds.has(itemId)) {
        continue
      }

      // Secondary check: content-based deduplication (for retrieved docs with different UUIDs)
      const source = item.metadata?.source || ''
      const contentKey = `${source}:::${item.pageContent.substring(0, 500)}`
      if (existingContentKeys.has(contentKey)) {
        continue
      }

      // Add the document if it's truly unique
      // Ensure metadata.uuid is set for deduplication consistency, and preserve the id field
      const newDoc =
        itemId === item.metadata?.uuid && itemId === item.id
          ? item
          : new Document({
              pageContent: item.pageContent,
              metadata: { ...item.metadata, uuid: itemId },
              id: itemId, // Preserve the id field from vector DB
            })

      newList.push(newDoc)
      existingIds.add(itemId)
      existingContentKeys.add(contentKey)
    } else if (typeof item === 'object' && item !== null) {
      // Plain object with pageContent
      const metadata = item.metadata || {}
      const pageContent = item.pageContent || ''
      // Use existing id from object if available (from vector DB), otherwise check metadata.uuid, fallback to generating new UUID
      let itemId = (item as any).id || metadata.uuid

      if (!itemId) {
        itemId = uuidv4()
      }

      if (!existingIds.has(itemId)) {
        newList.push(
          new Document({
            pageContent,
            metadata: { ...metadata, uuid: itemId },
            id: itemId, // Preserve the id field from vector DB
          }),
        )
        existingIds.add(itemId)
      }
    }
  }

  return [...existingList, ...newList]
}
