/**
 * Utility Functions
 *
 * General-purpose utility functions used throughout the application.
 */

import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

// ============================================================================
// Styling Utilities
// ============================================================================

/**
 * Combine and merge Tailwind CSS classes.
 *
 * Uses `clsx` to handle conditional classes and `tailwind-merge` to properly
 * merge conflicting Tailwind classes (e.g., "px-2 px-4" → "px-4").
 *
 * @param inputs - Any number of class values (strings, objects, arrays)
 * @returns Merged and deduplicated className string
 *
 * @example
 * cn("px-2 py-1", "px-4") // → "py-1 px-4"
 * cn("text-red-500", isActive && "text-blue-500") // → "text-blue-500" (if isActive)
 * cn({ "bg-gray-100": isHovered, "bg-white": !isHovered }) // Conditional classes
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

// ============================================================================
// Re-exports
// ============================================================================

// Chat utilities
export * from "./chat"

// String utilities
export * from "./string"

// Logger
export { logger } from "./logger"
