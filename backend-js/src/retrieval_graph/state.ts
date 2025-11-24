/**
 * State management for the retrieval graph.
 *
 * This module defines the state structures used in the retrieval graph. It includes
 * definitions for agent state, input state, and reducer functions.
 */

import { BaseMessage } from '@langchain/core/messages'
import { Document } from '@langchain/core/documents'
import { Annotation, messagesStateReducer } from '@langchain/langgraph'
import { reduceDocs } from '../utils.js'

/**
 * InputState represents the input to the agent.
 *
 * This is a restricted version of the State that is used to define
 * a narrower interface to the outside world vs. what is maintained internally.
 */
export const InputStateAnnotation = Annotation.Root({
  /**
   * Messages track the primary execution state of the agent.
   *
   * Typically accumulates a pattern of Human/AI/Human/AI messages.
   * Uses messagesStateReducer to merge messages by ID.
   */
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
})

/**
 * AgentState is the primary state of the retrieval agent.
 *
 * It extends InputState with additional internal state for research planning
 * and document retrieval.
 */
export const AgentStateAnnotation = Annotation.Root({
  /**
   * Messages track the conversation history.
   */
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),

  /**
   * A list of steps in the research plan.
   */
  steps: Annotation<string[]>({
    reducer: (existing, update) => {
      if (update === null || update === undefined) {
        return existing || []
      }
      return update
    },
    default: () => [],
  }),

  /**
   * Documents retrieved by the researcher.
   * Uses custom reducer to handle document deduplication.
   */
  documents: Annotation<Document[], Document[] | 'delete'>({
    reducer: reduceDocs,
    default: () => [],
  }),

  /**
   * Final answer. Useful for evaluations.
   */
  answer: Annotation<string>({
    reducer: (existing, update) => update || existing || '',
    default: () => '',
  }),

  /**
   * The original query from the user.
   */
  query: Annotation<string>({
    reducer: (existing, update) => update || existing || '',
    default: () => '',
  }),
})

// Type exports for use in other modules
export type InputState = typeof InputStateAnnotation.State
export type AgentState = typeof AgentStateAnnotation.State
