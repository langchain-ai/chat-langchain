/**
 * Metadata Types
 *
 * Type definitions for usage tracking and metadata.
 */

/**
 * Token usage and cost tracking for AI model calls.
 * Retrieved from LangSmith run metadata.
 */
export interface UsageMetadata {
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
  input_cost?: number
  output_cost?: number
  total_cost?: number
}

