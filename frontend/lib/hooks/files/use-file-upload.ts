/**
 * File Upload Hook
 *
 * Custom hook for managing file uploads (images, code files, logs, etc.)
 * - Handles drag & drop, paste, and file selection
 * - Validates file types and sizes
 * - Converts files to base64 for API transmission
 * - Manages upload errors and loading states
 */

import { useState, useCallback } from "react"
import type { ImageAttachment } from "../../types"
import { createImageAttachment, validateImageFile } from "../../utils/chat"
import {
  FILE_TOO_LARGE_MESSAGE,
  IMAGE_UNSUPPORTED_MODEL_MESSAGE,
  MAX_INPUT_CHARS,
} from "../../constants/features"

// ============================================================================
// Types
// ============================================================================

export interface UseFileUploadReturn {
  attachedFiles: ImageAttachment[]
  uploadError: string | null
  isDragging: boolean
  handleFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>
  handlePaste: (event: React.ClipboardEvent) => Promise<void>
  handleDrop: (event: React.DragEvent) => Promise<void>
  handleDragOver: (event: React.DragEvent) => void
  handleDragLeave: (event: React.DragEvent) => void
  removeFile: (fileId: string) => void
  clearFiles: () => void
  setUploadError: (error: string | null) => void
}

interface UseFileUploadOptions {
  getInputLength?: () => number
  disableImageUploads?: boolean
}

const isImageFile = (file: File): boolean =>
  file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp)$/i.test(file.name)

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook to manage file uploads with drag & drop, paste, and file selection support.
 *
 * @returns File upload state and handlers
 *
 * @example
 * ```tsx
 * const { attachedFiles, handleFileSelect, handlePaste, handleDrop, removeFile } = useFileUpload()
 *
 * return (
 *   <div onDrop={handleDrop} onPaste={handlePaste}>
 *     <input type="file" onChange={handleFileSelect} />
 *     {attachedFiles.map(file => (
 *       <FilePreview key={file.id} file={file} onRemove={removeFile} />
 *     ))}
 *   </div>
 * )
 * ```
 */
export function useFileUpload(options: UseFileUploadOptions = {}): UseFileUploadReturn {
  const [attachedFiles, setAttachedFiles] = useState<ImageAttachment[]>([])
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const getInputLength = options.getInputLength ?? (() => 0)
  const disableImageUploads = options.disableImageUploads ?? false

  /**
   * Process multiple files and add them to attached files list.
   * Validates each file and converts to base64.
   */
  const processFiles = useCallback(async (files: File[]) => {
    setUploadError(null)
    let acceptedTextLength = attachedFiles.reduce((total, file) => {
      if (file.mimeType?.startsWith('image/')) return total
      return total + (file.textLength ?? 0)
    }, 0)

    for (const file of files) {
      // Validate file
      const validation = validateImageFile(file)
      if (!validation.valid) {
        setUploadError(validation.error || "Invalid file")
        continue
      }

      try {
        const isImage = isImageFile(file)
        let textLength: number | undefined

        if (isImage && disableImageUploads) {
          setUploadError(IMAGE_UNSUPPORTED_MODEL_MESSAGE)
          continue
        }

        if (!isImage) {
          const text = await file.text()
          textLength = text.length

          if (textLength > MAX_INPUT_CHARS) {
            setUploadError(FILE_TOO_LARGE_MESSAGE)
            continue
          }

          if (getInputLength() + acceptedTextLength + textLength > MAX_INPUT_CHARS) {
            setUploadError(FILE_TOO_LARGE_MESSAGE)
            continue
          }
        }

        // Convert to attachment
        const imageAttachment = await createImageAttachment(file)
        imageAttachment.textLength = textLength
        setAttachedFiles(prev => [...prev, imageAttachment])
        acceptedTextLength += textLength ?? 0
      } catch (error) {
        console.error("Error processing file:", error)
        setUploadError("Failed to process file")
      }
    }
  }, [attachedFiles, disableImageUploads, getInputLength])

  /**
   * Handle file selection from input element.
   */
  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    await processFiles(Array.from(files))

    // Reset file input
    event.target.value = ""
  }, [processFiles])

  /**
   * Handle paste events (images from clipboard).
   */
  const handlePaste = useCallback(async (event: React.ClipboardEvent) => {
    const items = event.clipboardData?.items
    if (!items) return

    setUploadError(null)

    // Process clipboard items
    for (const item of Array.from(items)) {
      // Check if item is an image (paste only supports images for now)
      if (item.type.startsWith('image/')) {
        event.preventDefault() // Prevent default paste behavior for images

        if (disableImageUploads) {
          setUploadError(IMAGE_UNSUPPORTED_MODEL_MESSAGE)
          continue
        }

        const file = item.getAsFile()
        if (!file) continue

        // Validate file
        const validation = validateImageFile(file)
        if (!validation.valid) {
          setUploadError(validation.error || "Invalid image")
          continue
        }

        try {
          // Convert to image attachment
          const imageAttachment = await createImageAttachment(file)
          setAttachedFiles(prev => [...prev, imageAttachment])
          console.log('Pasted image from clipboard:', file.name || 'screenshot')
        } catch (error) {
          console.error("Error processing pasted image:", error)
          setUploadError("Failed to process pasted image")
        }
      }
    }
  }, [disableImageUploads])

  /**
   * Handle drag over event.
   */
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragging(true)
  }, [])

  /**
   * Handle drag leave event.
   */
  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragging(false)
  }, [])

  /**
   * Handle drop event.
   */
  const handleDrop = useCallback(async (event: React.DragEvent) => {
    event.preventDefault()
    setIsDragging(false)
    setUploadError(null)

    const files = event.dataTransfer?.files
    if (!files || files.length === 0) return

    await processFiles(Array.from(files))
  }, [processFiles])

  /**
   * Remove a file from the attached files list.
   */
  const removeFile = useCallback((fileId: string) => {
    setAttachedFiles(prev => {
      const file = prev.find(f => f.id === fileId)
      // Revoke object URL to free memory
      if (file?.url) {
        URL.revokeObjectURL(file.url)
      }
      return prev.filter(f => f.id !== fileId)
    })
  }, [])

  /**
   * Clear all attached files.
   * Note: We don't revoke URLs here because the message that was just sent
   * still needs them for rendering. URLs will be cleaned up by the browser
   * when the page is closed or refreshed.
   */
  const clearFiles = useCallback(() => {
    setAttachedFiles([])
    setUploadError(null)
  }, [])

  return {
    attachedFiles,
    uploadError,
    isDragging,
    handleFileSelect,
    handlePaste,
    handleDrop,
    handleDragOver,
    handleDragLeave,
    removeFile,
    clearFiles,
    setUploadError,
  }
}
