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
  OLLAMA_BASE_URL,
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
      'https://docs.langchain.com/oss/javascript/langchain/mcp',
      'https://docs.langchain.com/oss/javascript/langchain/agents',
      'https://docs.langchain.com/oss/javascript/langchain/context-engineering',
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
 * Main ingestion function.
 * Loads documents, splits them, and indexes them in Weaviate.
 */
export async function ingestDocs(): Promise<void> {
  console.log('Starting document ingestion...')

  // Chunks for nomic-embed-text (2K token context window)
  // Reduce to 2000 chars ≈ 500-650 tokens to avoid context overflow
  // TypeScript Ollama client seems more strict than Python version
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 4000,
    chunkOverlap: 200,
  })

  const embedding = getEmbeddingsModel(undefined, OLLAMA_BASE_URL)

  if (!embedding) {
    throw new Error('Embeddings model is required for ingestion')
  }

  const weaviateClient = await getWeaviateClient(
    WEAVIATE_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_API_KEY,
  )

  // Initialize PostgreSQL record manager
  let recordManager: PostgresRecordManager | undefined

  try {
    // Load documents
    console.log('Loading documents...')
    console.log('Step 1/5: Fetching documents from sitemap...')
    const generalGuidesAndTutorialsDocs =
      await ingestGeneralGuidesAndTutorials()
    console.log(
      `✓ Loaded ${generalGuidesAndTutorialsDocs.length} documents successfully`,
    )

    // Write raw documents to JSON file for inspection
    console.log('Step 2/5: Writing raw documents to file...')
    const rawDocsData = generalGuidesAndTutorialsDocs.map((doc) => {
      // Start with basic fields (matching Python format)
      const docData: Record<string, any> = {
        page_content: doc.pageContent,
        metadata: doc.metadata,
        type: 'Document',
      }

      // Add any additional fields from the document object
      // This matches the Python version which iterates through all attributes
      for (const key of Object.keys(doc)) {
        if (
          !['pageContent', 'metadata'].includes(key) &&
          !key.startsWith('_')
        ) {
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
    })

    const rawDocsFilePath = path.join(process.cwd(), '..', 'raw_docs_js.json')
    console.log(`Writing to: ${rawDocsFilePath}`)
    await fs.writeFile(
      rawDocsFilePath,
      JSON.stringify(rawDocsData, null, 2),
      'utf-8',
    )
    console.log(
      `✓ Wrote ${rawDocsData.length} raw documents to raw_docs_js.json`,
    )

    // Split documents
    console.log('Step 3/5: Splitting documents into chunks...')
    let docsTransformed = await textSplitter.splitDocuments(
      generalGuidesAndTutorialsDocs,
    )
    console.log(`Created ${docsTransformed.length} chunks (before filtering)`)

    // Filter out very short documents
    const beforeFilter = docsTransformed.length
    docsTransformed = docsTransformed.filter(
      (doc) => doc.pageContent.length > 10,
    )
    console.log(
      `✓ Filtered to ${docsTransformed.length} chunks (removed ${
        beforeFilter - docsTransformed.length
      } short chunks)`,
    )

    // Write transformed chunks to JSON file for inspection
    console.log('Step 4/5: Writing chunks to file...')
    const chunksData = docsTransformed.map((doc) => {
      // Start with basic fields
      const docData: Record<string, any> = {
        page_content: doc.pageContent,
        metadata: doc.metadata,
        type: 'Document',
      }

      // Add any additional fields from the document object
      // This matches the Python version which iterates through all attributes
      for (const key of Object.keys(doc)) {
        if (
          !['pageContent', 'metadata'].includes(key) &&
          !key.startsWith('_')
        ) {
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
    })

    const chunksFilePath = path.join(process.cwd(), '..', 'chunks_js.json')
    console.log(`Writing to: ${chunksFilePath}`)
    await fs.writeFile(
      chunksFilePath,
      JSON.stringify(chunksData, null, 2),
      'utf-8',
    )
    console.log(`✓ Wrote ${chunksData.length} chunks to chunks_js.json`)

    // Ensure required metadata fields exist
    // We try to return 'source' and 'title' metadata when querying vector store and
    // Weaviate will error at query time if one of the attributes is missing from a
    // retrieved document.
    console.log('Step 5/5: Ensuring metadata fields...')
    for (const doc of docsTransformed) {
      if (!doc.metadata.source) {
        doc.metadata.source = ''
      }
      if (!doc.metadata.title) {
        doc.metadata.title = ''
      }
    }
    console.log('✓ Metadata fields validated')

    // Create Weaviate store
    console.log('Indexing documents in Weaviate...')
    const store = new WeaviateStore(embedding, {
      client: weaviateClient,
      indexName: WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
      textKey: 'text',
      metadataKeys: ['source', 'title'],
    })

    // Initialize PostgreSQL record manager for tracking indexed documents
    // Remove sslmode from connection string as pg client doesn't parse it properly
    // and explicitly set ssl to false since the server doesn't support SSL
    const dbUrl = RECORD_MANAGER_DB_URL?.split('?')[0] || RECORD_MANAGER_DB_URL

    recordManager = new PostgresRecordManager(
      `weaviate/${WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME}`,
      {
        postgresConnectionOptions: {
          connectionString: dbUrl,
          // Explicitly disable SSL (server doesn't support it)
          ssl: false,
        },
      },
    )

    // Create schema for record manager
    await recordManager.createSchema()
    console.log('Record manager schema created')

    // Use index function for smart incremental indexing
    // This prevents duplicates and handles updates/deletions
    const indexingStats = await index({
      docsSource: docsTransformed,
      recordManager,
      vectorStore: store,
      options: {
        cleanup: 'full',
        sourceIdKey: 'source',
        forceUpdate:
          (process.env.FORCE_UPDATE || 'false').toLowerCase() === 'true',
      },
    })

    console.log(`Indexing stats:`, indexingStats)

    // Get total count
    const collection = await weaviateClient.collections.get(
      WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
    )
    const totalCount = await collection.aggregate.overAll()
    console.log(`Total vectors in collection: ${totalCount.totalCount}`)

    console.log('Document ingestion completed successfully!')
  } finally {
    // Close connections
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
