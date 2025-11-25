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
 * Shared channel definitions to ensure all annotations use the same channel instances.
 * This prevents "Channel already exists with a different type" errors.
 */
const messagesChannel = Annotation<BaseMessage[]>({
  reducer: messagesStateReducer,
  default: () => [],
})

const documentsChannel = Annotation<Document[], Document[] | 'delete'>({
  reducer: reduceDocs,
  default: () => [],
})

const answerChannel = Annotation<string>({
  reducer: (existing, update) => {
    // Always return a string value for LangSmith compatibility
    if (update !== undefined && update !== null) {
      return String(update)
    }
    if (existing !== undefined && existing !== null) {
      return String(existing)
    }
    return ''
  },
  default: () => '',
})

const queryChannel = Annotation<string>({
  reducer: (existing, update) => {
    // Always return a string value for LangSmith compatibility
    if (update !== undefined && update !== null) {
      return String(update)
    }
    if (existing !== undefined && existing !== null) {
      return String(existing)
    }
    return ''
  },
  default: () => '',
})

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
  messages: messagesChannel,
})

/**
 * AgentState is the primary state of the retrieval agent.
 *
 * It extends InputState with additional internal state for research planning
 * and document retrieval, matching Python's class AgentState(InputState) pattern.
 */
export const AgentStateAnnotation = Annotation.Root({
  /**
   * Inherit messages from InputStateAnnotation
   */
  ...InputStateAnnotation.spec,

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
   */
  documents: documentsChannel,

  /**
   * Final answer. Useful for evaluations.
   */
  answer: answerChannel,

  /**
   * The original query from the user.
   */
  query: queryChannel,
})

// Type exports for use in other modules
export type InputState = typeof InputStateAnnotation.State
export type AgentState = typeof AgentStateAnnotation.State
