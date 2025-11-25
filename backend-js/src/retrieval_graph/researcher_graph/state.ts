/**
 * State management for the researcher subgraph.
 *
 * This module defines state structures for the researcher subgraph which
 * generates queries and retrieves relevant documents.
 */

import { Document } from '@langchain/core/documents'
import { Annotation } from '@langchain/langgraph'
import { reduceDocs } from '../../utils.js'

/**
 * ResearcherState manages the state for query generation and document retrieval.
 */
export const ResearcherStateAnnotation = Annotation.Root({
  /**
   * The question or research step to investigate.
   */
  question: Annotation<string>({
    reducer: (existing, update) => update || existing || '',
    default: () => '',
  }),

  /**
   * Generated search queries for this research step.
   */
  queries: Annotation<string[]>({
    reducer: (existing, update) => {
      if (update === null || update === undefined) {
        return existing || []
      }
      return update
    },
    default: () => [],
  }),

  /**
   * Documents retrieved from all queries.
   * Uses custom reducer to handle document deduplication.
   */
  documents: Annotation<Document[]>({
    reducer: reduceDocs,
    default: () => [],
  }),

  /**
   * Index of this query in the list of queries (for tracking).
   */
  queryIndex: Annotation<number>({
    reducer: (existing, update) => update ?? existing ?? 0,
    default: () => 0,
  }),
})

/**
 * QueryState represents the state for a single query retrieval task.
 */
export const QueryStateAnnotation = Annotation.Root({
  /**
   * The search query string.
   */
  query: Annotation<string>({
    reducer: (existing, update) => update || existing || '',
    default: () => '',
  }),

  /**
   * Index of this query in the list of queries (for tracking).
   */
  queryIndex: Annotation<number>({
    reducer: (existing, update) => update ?? existing ?? 0,
    default: () => 0,
  }),

  /**
   * Documents retrieved for this specific query.
   */
  documents: Annotation<Document[]>({
    reducer: reduceDocs,
    default: () => [],
  }),
})

// Type exports for use in other modules
export type ResearcherState = typeof ResearcherStateAnnotation.State
export type QueryState = typeof QueryStateAnnotation.State
