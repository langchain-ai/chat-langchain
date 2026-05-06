/**
 * File Validation Utilities
 *
 * Functions for validating files before upload.
 */

import { generateMessageId } from "./message-helpers"
import type { ImageAttachment } from "../../types"

/**
 * Convert a File object to a base64 string.
 * Used for encoding images before sending to the backend.
 */
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // Remove the data URL prefix (e.g., "data:image/png;base64,")
      const base64 = result.split(",")[1]
      resolve(base64)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/**
 * Create an ImageAttachment from a File object.
 * Converts the file to base64 and creates a preview URL.
 */
export const createImageAttachment = async (file: File): Promise<ImageAttachment> => {
  const base64 = await fileToBase64(file)
  const url = URL.createObjectURL(file)

  return {
    id: generateMessageId(),
    base64,
    url,
    mimeType: file.type,
    name: file.name,
    size: file.size,
  }
}

/**
 * Validate if a file is supported and within size limits.
 * Supports images, code files, logs, and text files.
 */
export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  // HAR files can be large network captures — allow up to 50MB.
  // Public CLC does not support HAR analysis; HAR files are ignored before streaming.
  // All other file types retain the original 10MB limit.
  const isHar = file.name.toLowerCase().endsWith(".har")
  const maxSize = isHar ? 50 * 1024 * 1024 : 10 * 1024 * 1024
  const maxSizeLabel = isHar ? "50MB" : "10MB"
  if (file.size > maxSize) {
    return { valid: false, error: `File must be smaller than ${maxSizeLabel}` }
  }

  // Supported mimetypes
  const supportedMimeTypes = [
    // Images
    "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp",
    // Text/Code
    "text/plain", "text/markdown", "text/x-python", "text/x-java",
    "text/x-c", "text/x-c++", "text/javascript", "text/typescript",
    "text/html", "text/css", "text/xml", "application/json",
    "application/javascript", "application/typescript",
    "application/x-python", "application/x-python-code",
    "application/x-sh", "text/x-sh", "text/x-log"
  ]

  // Supported file extensions (fallback if mimetype is not set correctly)
  const supportedExtensions = [
    // Images
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    // Code files
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bash",
    // Config/Data files
    ".yaml", ".yml", ".json", ".xml", ".html", ".css", ".md",
    // Text/Log files
    ".txt", ".log", ".sql", ".graphql",
    // Other languages
    ".r", ".swift", ".kt", ".scala",
    // Network/Debug files
    ".har"
  ]

  const fileName = file.name.toLowerCase()
  const hasValidExtension = supportedExtensions.some(ext => fileName.endsWith(ext))
  const hasValidMimetype = supportedMimeTypes.includes(file.type)

  // Accept if either mimetype or extension is valid
  if (!hasValidMimetype && !hasValidExtension) {
    const ext = fileName.split('.').pop()?.toLowerCase()
    return {
      valid: false,
      error: `Unsupported file type${ext ? ` (.${ext})` : ''}. Supported: images, code files, logs, configs (see button tooltip for full list)`
    }
  }

  return { valid: true }
}

