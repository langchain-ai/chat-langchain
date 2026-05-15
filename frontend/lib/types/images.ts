/**
 * Image Types
 *
 * Type definitions for image attachments.
 */

/**
 * Represents an image attachment in a message.
 */
export interface ImageAttachment {
  id: string
  url?: string // URL for displaying the image
  base64?: string // Base64-encoded image data
  mimeType: string // e.g., "image/png", "image/jpeg"
  name?: string // Original filename
  size?: number // File size in bytes
  textLength?: number // Character count for non-image text/code files
}

