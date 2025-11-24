/**
 * Base configuration for the agent.
 *
 * This module defines the base configuration parameters for indexing and retrieval operations.
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";

/**
 * Schema for backwards-compatible model name mapping
 */
const MODEL_NAME_TO_RESPONSE_MODEL: Record<string, string> = {
  anthropic_claude_3_5_sonnet: "anthropic/claude-3-5-sonnet-20240620",
};

/**
 * BaseConfiguration schema using Zod for validation
 */
export const BaseConfigurationSchema = z.object({
  /**
   * Name of the embedding model to use.
   * Must be a valid embedding model name.
   * @default "ollama/nomic-embed-text"
   */
  embeddingModel: z
    .string()
    .default(process.env.EMBEDDING_MODEL || "ollama/nomic-embed-text")
    .describe("Name of the embedding model to use"),

  /**
   * The vector store provider to use for retrieval.
   * @default "weaviate"
   */
  retrieverProvider: z
    .enum(["weaviate"])
    .default("weaviate")
    .describe("The vector store provider to use for retrieval"),

  /**
   * Additional keyword arguments to pass to the search function of the retriever.
   */
  searchKwargs: z
    .record(z.any())
    .default({})
    .describe("Additional keyword arguments for the retriever search function"),

  /**
   * The number of documents to retrieve (backwards compatibility).
   * Use searchKwargs instead.
   * @default 6
   */
  k: z.number().default(6).describe("Number of documents to retrieve"),
});

export type BaseConfiguration = z.infer<typeof BaseConfigurationSchema>;

/**
 * Update configurable parameters for backwards compatibility
 */
function updateConfigurableForBackwardsCompatibility(
  configurable: Record<string, any>
): Record<string, any> {
  const update: Record<string, any> = {};

  if ("k" in configurable) {
    update.searchKwargs = { k: configurable.k };
  }

  if ("model_name" in configurable) {
    update.responseModel =
      MODEL_NAME_TO_RESPONSE_MODEL[configurable.model_name] || configurable.model_name;
  }

  if (Object.keys(update).length > 0) {
    return { ...configurable, ...update };
  }

  return configurable;
}

/**
 * Extract configuration from RunnableConfig
 */
export function getBaseConfiguration(config?: RunnableConfig): BaseConfiguration {
  const configurable = config?.configurable || {};
  const updated = updateConfigurableForBackwardsCompatibility(configurable);

  // Convert snake_case to camelCase for embedding_model
  if ("embedding_model" in updated) {
    updated.embeddingModel = updated.embedding_model;
    delete updated.embedding_model;
  }
  if ("retriever_provider" in updated) {
    updated.retrieverProvider = updated.retriever_provider;
    delete updated.retriever_provider;
  }
  if ("search_kwargs" in updated) {
    updated.searchKwargs = updated.search_kwargs;
    delete updated.search_kwargs;
  }

  // Parse and validate with defaults
  return BaseConfigurationSchema.parse(updated);
}

