/**
 * Production-safe logging utility
 *
 * Logging behavior by environment:
 * - Development: info, warn, error
 * - Production: silent
 *
 * This keeps the public production deployment quiet.
 */

type LogLevel = "debug" | "info" | "warn" | "error"

class Logger {
  private isProduction: boolean
  private isSilent: boolean

  constructor() {
    this.isProduction = process.env.NODE_ENV === "production"
    this.isSilent = this.isProduction
  }

  /**
   * Debug logs are disabled in this public frontend.
   */
  debug(message: string, ...args: any[]) {
    return
  }

  /**
   * Info logs - shown in development, hidden in production
   */
  info(message: string, ...args: any[]) {
    if (this.isSilent) return
    console.log(`[INFO] ${message}`, ...args)
  }

  /**
   * Warning logs - hidden in production
   */
  warn(message: string, ...args: any[]) {
    if (this.isSilent) return
    console.warn(`[WARN] ${message}`)
  }

  /**
   * Error logs - hidden in production
   */
  error(message: string, error?: Error | unknown) {
    if (this.isSilent) return
    console.error(`[ERROR] ${message}`)
  }

  /**
   * Check if logging is enabled for a specific level
   */
  isEnabled(level: LogLevel): boolean {
    // External production: completely silent
    if (this.isSilent) return false

    return true
  }

  /**
   * Get current environment info
   */
  getEnvInfo() {
    return {
      deployment: "public",
      mode: this.isProduction ? "production" : "development",
      loggingEnabled: !this.isSilent,
      isSilent: this.isSilent,
    }
  }
}

// Export singleton instance
export const logger = new Logger()

if (typeof window !== "undefined") {
  const envInfo = logger.getEnvInfo()
  if (envInfo.mode === "development") {
    console.log(`[Logger] Initialized:`, envInfo)
  }
}
