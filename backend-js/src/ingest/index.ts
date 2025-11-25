/**
 * Load HTML from files, clean up, split, ingest into Weaviate.
 *
 * This module provides functions to load documents from sitemaps,
 * process them, and index them in Weaviate for retrieval.
 */

// Load environment variables from .env file
import 'dotenv/config'

import { promises as fs } from 'fs'
import * as path from 'path'
import { RecursiveUrlLoader } from '@langchain/community/document_loaders/web/recursive_url'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { Document } from '@langchain/core/documents'
import type { CheerioAPI } from 'cheerio'
import { WeaviateStore } from '@langchain/weaviate'
import { PostgresRecordManager } from '@langchain/community/indexes/postgres'
import { index } from '@langchain/core/indexing'
import { getWeaviateClient } from '../utils.js'
import { getEmbeddingsModel } from '../embeddings.js'
import { FixedSitemapLoader } from './FixedSitemapLoader.js'
import {
  OLLAMA_BASE_EMBEDDING_DOCS_URL,
  WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
} from '../constants.js'
import {
  langchainDocsExtractor,
  simpleExtractor,
  extractMetadata,
} from './parser.js'

const WEAVIATE_URL = process.env.WEAVIATE_URL
const WEAVIATE_GRPC_URL = process.env.WEAVIATE_GRPC_URL
const WEAVIATE_API_KEY = process.env.WEAVIATE_API_KEY
const RECORD_MANAGER_DB_URL = process.env.RECORD_MANAGER_DB_URL

/**
 * Load documents from a sitemap and extract content.
 *
 * @param sitemapUrl - URL of the sitemap to load
 * @param filterUrls - Array of URL patterns to filter (optional)
 * @param extractor - Function to extract content from HTML
 * @returns Array of loaded documents
 */
async function loadFromSitemap(
  sitemapUrl: string,
  filterUrls?: string[],
  extractor: (html: string | CheerioAPI) => string = simpleExtractor,
): Promise<Document[]> {
  console.log(`Loading documents from sitemap: ${sitemapUrl}`)

  // Use FixedSitemapLoader to load documents from sitemap
  // This fixes a bug in the original SitemapLoader where filterUrls has inverted logic
  // Convert filter URLs to regex patterns that match the URL
  // Escape special regex characters in URLs for literal matching
  const filterRegexes = filterUrls?.map((url) => {
    const escapedUrl = url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return new RegExp(escapedUrl)
  })

  const loader = new FixedSitemapLoader(sitemapUrl, {
    filterUrls: filterRegexes,
    chunkSize: 50, // Load in smaller chunks to avoid timeout
    extractor, // Pass the custom extractor
  })

  console.log('Loading documents from sitemap...')
  console.log(
    `Filter patterns: ${filterUrls?.join(', ') || 'none (loading all)'}`,
  )

  try {
    // FixedSitemapLoader now handles content extraction and metadata during load
    // It uses the provided extractor (or simpleExtractor by default) and extractMetadata internally
    const docs = await loader.load()
    console.log(`Successfully loaded and processed ${docs.length} documents`)
    return docs
  } catch (error) {
    console.error('Error loading from sitemap:', error)
    throw error
  }
}

/**
 * Load documents recursively from a base URL.
 *
 * @param baseUrl - Starting URL for recursive crawling
 * @param maxDepth - Maximum depth to crawl
 * @param extractor - Function to extract content from HTML
 * @param excludeDirs - Directories to exclude from crawling
 * @returns Array of loaded documents
 */
async function loadFromRecursiveUrl(
  baseUrl: string,
  maxDepth: number = 2,
  extractor: (html: string | CheerioAPI) => string = simpleExtractor,
  excludeDirs?: string[],
): Promise<Document[]> {
  console.log(`Loading documents recursively from: ${baseUrl}`)

  const loader = new RecursiveUrlLoader(baseUrl, {
    maxDepth,
    extractor,
    excludeDirs,
    preventOutside: true,
    timeout: 10000,
  })

  console.log('Loading documents recursively...')
  const docs = await loader.load()
  console.log(`Loaded ${docs.length} documents recursively`)

  // Process documents to extract metadata
  const documents: Document[] = []

  for (const doc of docs) {
    try {
      // Extract additional metadata from the original HTML if needed
      const metadata = extractMetadata(doc.pageContent)

      documents.push(
        new Document({
          pageContent: doc.pageContent,
          metadata: {
            ...doc.metadata,
            ...metadata,
          },
        }),
      )

      console.log(`Processed: ${doc.metadata.source}`)
    } catch (error) {
      console.error(`Failed to process document:`, error)
    }
  }

  return documents
}

/**
 * Load LangChain Python docs (to be deprecated once docs are migrated).
 */
async function loadLangchainPythonDocs(): Promise<Document[]> {
  return loadFromSitemap(
    'https://python.langchain.com/sitemap.xml',
    ['https://python.langchain.com/'],
    langchainDocsExtractor,
  )
}

/**
 * Load LangChain JS docs (to be deprecated once docs are migrated).
 */
async function loadLangchainJsDocs(): Promise<Document[]> {
  return loadFromSitemap(
    'https://js.langchain.com/sitemap.xml',
    ['https://js.langchain.com/docs/concepts'],
    simpleExtractor,
  )
}

/**
 * Load from aggregated docs site.
 */
async function loadAggregatedDocsSite(): Promise<Document[]> {
  console.log('Loading from aggregated docs site...')
  const docs = await loadFromSitemap(
    'https://docs.langchain.com/sitemap.xml',
    [
      // 'https://docs.langchain.com/oss/javascript',
      'https://docs.langchain.com/oss/javascript/langchain/mcp',
      'https://docs.langchain.com/oss/javascript/langchain/agents',
      'https://docs.langchain.com/oss/javascript/langchain/context-engineering',
      'https://docs.langchain.com/oss/javascript/concepts/context',
    ],
    simpleExtractor,
  )

  if (docs.length === 0) {
    console.warn(
      'WARNING: No documents matched the filter criteria! Check your filter URLs.',
    )
  }

  return docs
}

/**
 * Ingest general guides and tutorials.
 */
async function ingestGeneralGuidesAndTutorials(): Promise<Document[]> {
  const aggregatedSiteDocs = await loadAggregatedDocsSite()

  if (aggregatedSiteDocs.length === 0) {
    throw new Error(
      'No documents were loaded! Check your sitemap URL and filter patterns.',
    )
  }

  return aggregatedSiteDocs
}

/**
 * Serialize a document to JSON format for file output.
 *
 * @param doc - Document to serialize
 * @returns Serialized document data
 */
function serializeDocumentForJson(doc: Document): Record<string, any> {
  // Start with basic fields (matching Python format)
  const docData: Record<string, any> = {
    page_content: doc.pageContent,
    metadata: doc.metadata,
    type: 'Document',
  }

  // Add any additional fields from the document object
  // This matches the Python version which iterates through all attributes
  for (const key of Object.keys(doc)) {
    if (!['pageContent', 'metadata'].includes(key) && !key.startsWith('_')) {
      const value = (doc as any)[key]
      // Only include serializable values
      if (
        value !== undefined &&
        typeof value !== 'function' &&
        typeof value !== 'symbol'
      ) {
        try {
          JSON.stringify(value)
          docData[key] = value
        } catch {
          // Skip non-serializable values
        }
      }
    }
  }

  return docData
}

/**
 * Write documents to a JSON file for inspection.
 *
 * @param documents - Documents to write
 * @param filename - Name of the output file
 * @param description - Description for logging
 */
async function writeDocumentsToJsonFile(
  documents: Document[],
  filename: string,
  description: string,
): Promise<void> {
  const serializedData = documents.map(serializeDocumentForJson)
  const filePath = path.join(process.cwd(), '..', filename)

  console.log(`Writing to: ${filePath}`)
  await fs.writeFile(filePath, JSON.stringify(serializedData, null, 2), 'utf-8')
  console.log(`✓ Wrote ${serializedData.length} ${description} to ${filename}`)
}

/**
 * Split documents into chunks and filter out short ones.
 *
 * @param documents - Documents to split
 * @param textSplitter - Text splitter instance
 * @param minLength - Minimum content length to keep
 * @returns Filtered chunks
 */
async function splitAndFilterDocuments(
  documents: Document[],
  textSplitter: RecursiveCharacterTextSplitter,
  minLength: number = 10,
): Promise<Document[]> {
  console.log('Step 3/5: Splitting documents into chunks...')
  let chunks = await textSplitter.splitDocuments(documents)
  console.log(`Created ${chunks.length} chunks (before filtering)`)

  // Filter out very short documents
  const beforeFilter = chunks.length
  chunks = chunks.filter((doc) => doc.pageContent.length > minLength)
  console.log(
    `✓ Filtered to ${chunks.length} chunks (removed ${
      beforeFilter - chunks.length
    } short chunks)`,
  )

  return chunks
}

/**
 * Ensure required metadata fields exist in all documents.
 * Weaviate will error at query time if required attributes are missing.
 *
 * @param documents - Documents to validate
 */
function ensureRequiredMetadata(documents: Document[]): void {
  console.log('Step 5/5: Ensuring metadata fields...')

  for (const doc of documents) {
    if (!doc.metadata.source) {
      doc.metadata.source = ''
    }
    if (!doc.metadata.title) {
      doc.metadata.title = ''
    }
  }

  console.log('✓ Metadata fields validated')
}

/**
 * Create a Weaviate vector store instance.
 *
 * @param weaviateClient - Weaviate client
 * @param embedding - Embeddings model
 * @returns Weaviate store instance
 */
function createWeaviateVectorStore(
  weaviateClient: any,
  embedding: any,
): WeaviateStore {
  console.log('Indexing documents in Weaviate...')

  return new WeaviateStore(embedding, {
    client: weaviateClient,
    indexName: WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
    textKey: 'text',
    metadataKeys: ['source', 'title'],
  })
}

/**
 * Create and initialize PostgreSQL record manager.
 *
 * @returns Initialized record manager
 */
async function createRecordManager(): Promise<PostgresRecordManager> {
  // Remove sslmode from connection string as pg client doesn't parse it properly
  // and explicitly set ssl to false since the server doesn't support SSL
  const dbUrl = RECORD_MANAGER_DB_URL?.split('?')[0] || RECORD_MANAGER_DB_URL

  const recordManager = new PostgresRecordManager(
    `weaviate/${WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME}`,
    {
      postgresConnectionOptions: {
        connectionString: dbUrl,
        ssl: false,
      },
    },
  )

  await recordManager.createSchema()
  console.log('Record manager schema created')

  return recordManager
}

/**
 * Index documents in the vector store with record manager tracking.
 *
 * @param documents - Documents to index
 * @param vectorStore - Vector store instance
 * @param recordManager - Record manager instance
 * @returns Indexing statistics
 */
async function indexDocumentsInVectorStore(
  documents: Document[],
  vectorStore: WeaviateStore,
  recordManager: PostgresRecordManager,
): Promise<any> {
  const indexingStats = await index({
    docsSource: documents,
    recordManager,
    vectorStore,
    options: {
      cleanup: 'full',
      sourceIdKey: 'source',
      forceUpdate:
        (process.env.FORCE_UPDATE || 'false').toLowerCase() === 'true',
    },
  })

  console.log(`Indexing stats:`, indexingStats)
  return indexingStats
}

/**
 * Get and log total vector count from Weaviate collection.
 *
 * @param weaviateClient - Weaviate client
 */
async function logTotalVectorCount(weaviateClient: any): Promise<void> {
  const collection = await weaviateClient.collections.get(
    WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
  )
  const totalCount = await collection.aggregate.overAll()
  console.log(`Total vectors in collection: ${totalCount.totalCount}`)
}

/**
 * Main ingestion function.
 * Orchestrates the document ingestion pipeline: load, split, and index documents.
 */
export async function ingestDocs(): Promise<void> {
  console.log('Starting document ingestion...')

  // Initialize text splitter
  // Chunks for nomic-embed-text (2K token context window)
  // Reduce to 2000 chars ≈ 500-650 tokens to avoid context overflow
  // TypeScript Ollama client seems more strict than Python version
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 4000,
    chunkOverlap: 200,
  })

  // Initialize embeddings model
  const embedding = getEmbeddingsModel(
    undefined,
    OLLAMA_BASE_EMBEDDING_DOCS_URL,
  )
  if (!embedding) {
    throw new Error('Embeddings model is required for ingestion')
  }

  // Initialize Weaviate client
  const weaviateClient = await getWeaviateClient(
    WEAVIATE_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_API_KEY,
  )

  let recordManager: PostgresRecordManager | undefined

  try {
    // Step 1: Load documents
    console.log('Loading documents...')
    console.log('Step 1/5: Fetching documents from sitemap...')
    const rawDocuments = await ingestGeneralGuidesAndTutorials()
    console.log(`✓ Loaded ${rawDocuments.length} documents successfully`)

    // Step 2: Write raw documents to file
    console.log('Step 2/5: Writing raw documents to file...')
    await writeDocumentsToJsonFile(
      rawDocuments,
      'raw_docs_js.json',
      'raw documents',
    )

    // Step 3: Split and filter documents
    const chunks = await splitAndFilterDocuments(rawDocuments, textSplitter)

    // Step 4: Write chunks to file
    console.log('Step 4/5: Writing chunks to file...')
    await writeDocumentsToJsonFile(chunks, 'chunks_js.json', 'chunks')

    // Step 5: Ensure required metadata
    ensureRequiredMetadata(chunks)

    // Create vector store and record manager
    const vectorStore = createWeaviateVectorStore(weaviateClient, embedding)
    recordManager = await createRecordManager()

    // Index documents
    await indexDocumentsInVectorStore(chunks, vectorStore, recordManager)

    // Log final count
    await logTotalVectorCount(weaviateClient)

    console.log('Document ingestion completed successfully!')
  } finally {
    // Cleanup connections
    if (recordManager) {
      await recordManager.end()
      console.log('Record manager connection closed')
    }
    await weaviateClient.close()
    console.log('Weaviate client closed')
  }
}

/**
 * Run ingestion if this file is executed directly
 */
if (import.meta.url === `file://${process.argv[1]}`) {
  ingestDocs()
    .then(() => {
      console.log('Done!')
      process.exit(0)
    })
    .catch((error) => {
      console.error('Ingestion failed:', error)
      process.exit(1)
    })
}
