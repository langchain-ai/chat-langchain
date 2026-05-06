/**
 * Production-safe logging utility
 *
 * Logging behavior by environment:
 * - Internal + Development: All logs (debug, info, warn, error)
 * - Internal + Production: Info, warn, error (no debug)
 * - External + Development: Info, warn, error (for testing)
 * - External + Production: NOTHING (completely silent)
 *
 * This ensures external production deployments have zero console logs.
 */

import { getDeploymentEnv } from "../config/deployment-config"

type LogLevel = "debug" | "info" | "warn" | "error"

class Logger {
  private isExternal: boolean
  private isProduction: boolean
  private isSilent: boolean

  constructor() {
    this.isExternal = getDeploymentEnv() === "external"
    this.isProduction = process.env.NODE_ENV === "production"
    // Completely silent in external production
    this.isSilent = this.isExternal && this.isProduction
  }

  /**
   * Debug logs - only shown in internal development
   */
  debug(message: string, ...args: any[]) {
    if (this.isSilent) return
    if (!this.isExternal && !this.isProduction) {
      console.log(`[DEBUG] ${message}`, ...args)
    }
  }

  /**
   * Info logs - shown in development, hidden in external production
   */
  info(message: string, ...args: any[]) {
    if (this.isSilent) return
    if (!this.isProduction || !this.isExternal) {
      console.log(`[INFO] ${message}`, ...args)
    }
  }

  /**
   * Warning logs - hidden in external production
   */
  warn(message: string, ...args: any[]) {
    if (this.isSilent) return
    if (this.isExternal && this.isProduction) return

    if (this.isExternal) {
      // External dev: sanitize args
      console.warn(`[WARN] ${message}`)
    } else {
      console.warn(`[WARN] ${message}`, ...args)
    }
  }

  /**
   * Error logs - hidden in external production (use error reporting service instead)
   */
  error(message: string, error?: Error | unknown) {
    if (this.isSilent) return

    if (this.isExternal && this.isProduction) {
      // External production: completely silent
      // TODO: Send to error monitoring service (Sentry, Datadog, etc.)
      return
    }

    if (this.isExternal) {
      // External dev: sanitized errors
      console.error(`[ERROR] ${message}`)
    } else {
      // Internal: full error details
      console.error(`[ERROR] ${message}`, error)
    }
  }

  /**
   * Check if logging is enabled for a specific level
   */
  isEnabled(level: LogLevel): boolean {
    // External production: completely silent
    if (this.isSilent) return false

    if (this.isExternal && this.isProduction) {
      return false
    }
    if (this.isExternal) {
      // External dev: info, warn, error only
      return level !== "debug"
    }
    if (this.isProduction) {
      // Internal production: info and above
      return level !== "debug"
    }
    // Development: all levels
    return true
  }

  /**
   * Get current environment info
   */
  getEnvInfo() {
    return {
      deployment: this.isExternal ? "external" : "internal",
      mode: this.isProduction ? "production" : "development",
      loggingEnabled: !this.isSilent,
      isSilent: this.isSilent,
    }
  }
}

// Export singleton instance
export const logger = new Logger()

// Log initialization info only in internal development (never in production or external)
if (typeof window !== "undefined") {
  const envInfo = logger.getEnvInfo()
  // Only show in internal development
  if (envInfo.deployment === "internal" && envInfo.mode === "development") {
    console.log(`[Logger] Initialized:`, envInfo)
  }
}
