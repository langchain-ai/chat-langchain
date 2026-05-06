/**
 * API Constants
 *
 * Configuration constants for API endpoints and keys.
 */

/**
 * Get LangGraph API URL based on deployment environment and local/remote preference.
 * 
 * Supports both local and remote deployments for internal and external environments:
 * - NEXT_PUBLIC_USE_LOCAL_DEPLOYMENT=true → uses local URLs (default: false)
 * - Internal deployment → uses INTERNAL_LOCAL or INTERNAL_REMOTE
 * - External deployment → uses EXTERNAL_LOCAL or EXTERNAL_REMOTE
 * 
 * Environment variables:
 * - NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL_LOCAL (default: http://127.0.0.1:2024)
 * - NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL_REMOTE
 * - NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL_LOCAL (default: http://127.0.0.1:2024)
 * - NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL_REMOTE
 * - NEXT_PUBLIC_USE_LOCAL_DEPLOYMENT (default: false)
 */
function getLangGraphApiUrl(): string {
  const deploymentEnv = process.env.NEXT_PUBLIC_DEPLOYMENT_ENV || "external"
  const useLocal = process.env.NEXT_PUBLIC_USE_LOCAL_DEPLOYMENT !== 'false'

  if (deploymentEnv === "external") {
    // External deployment
    const url = useLocal
      ? process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL_LOCAL || 'http://127.0.0.1:2024'
      : process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL_REMOTE || process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL || 'http://127.0.0.1:2024'
    
    if (!url) {
      throw new Error(
        useLocal
          ? 'NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL_LOCAL is not defined for external local deployment'
          : 'NEXT_PUBLIC_LANGGRAPH_API_URL_EXTERNAL_REMOTE is not defined for external remote deployment'
      )
    }
    // Only log in development, never in production
    if (process.env.NODE_ENV === 'development') {
      console.info(`[LangGraph] External ${useLocal ? 'local' : 'remote'} deployment → routing to:`, url)
    }
    return url
  } else {
    // Internal deployment (default)
    const url = useLocal
      ? process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL_LOCAL || 'http://127.0.0.1:2024'
      : process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL_REMOTE || process.env.NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL || 'http://127.0.0.1:2024'
    
    if (!url) {
      throw new Error(
        useLocal
          ? 'NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL_LOCAL is not defined for internal local deployment'
          : 'NEXT_PUBLIC_LANGGRAPH_API_URL_INTERNAL_REMOTE is not defined for internal remote deployment'
      )
    }
    // Only log in development, never in production
    if (process.env.NODE_ENV === 'development') {
      console.info(`[LangGraph] Internal ${useLocal ? 'local' : 'remote'} deployment → routing to:`, url)
    }
    return url
  }
}

export const LANGGRAPH_API_URL = getLangGraphApiUrl()

// LangGraph Server API key (used for LangGraph client, not LangSmith)
// Note: This should be undefined in browser for security
// LangGraph Cloud deployments need auth disabled or custom auth configured
export const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY

