/**
 * File Upload Hook
 *
 * Custom hook for managing file uploads (images, code files, logs, etc.)
 * - Handles drag & drop, paste, and file selection
 * - Validates file types and sizes
 * - Converts files to base64 for API transmission
 * - Manages upload errors and loading states
 */

import { useState, useCallback, useEffect, useRef } from "react"
import type { ImageAttachment } from "../../types"
import { createImageAttachment, validateImageFile } from "../../utils/chat"
import { appendWithinInputBudget, getAttachmentMimeType } from "../../utils/chat/composer-state"
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
  getAttachmentMimeType(file).startsWith("image/")

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
  const attachedFilesRef = useRef<ImageAttachment[]>([])
  const uploadGenerationRef = useRef(0)
  const getInputLengthRef = useRef(options.getInputLength)
  getInputLengthRef.current = options.getInputLength
  const disableImageUploads = options.disableImageUploads ?? false

  useEffect(() => () => {
    uploadGenerationRef.current += 1
  }, [])

  /**
   * Process multiple files and add them to attached files list.
   * Validates each file and converts to base64.
   */
  const processFiles = useCallback(async (files: File[]) => {
    const generation = uploadGenerationRef.current
    setUploadError(null)

    for (const file of files) {
      if (generation !== uploadGenerationRef.current) return
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
          if (generation !== uploadGenerationRef.current) return
          textLength = text.length

          if (textLength > MAX_INPUT_CHARS) {
            setUploadError(FILE_TOO_LARGE_MESSAGE)
            continue
          }
        }

        // Convert to attachment
        const imageAttachment = await createImageAttachment(file)
        imageAttachment.textLength = textLength
        if (generation !== uploadGenerationRef.current) {
          if (imageAttachment.url) URL.revokeObjectURL(imageAttachment.url)
          return
        }

        // Concurrent picker/drop operations and typing can change the budget
        // while FileReader is working. Admit against the latest state atomically.
        const nextFiles = appendWithinInputBudget(
          attachedFilesRef.current,
          imageAttachment,
          getInputLengthRef.current?.() ?? 0,
          MAX_INPUT_CHARS,
        )
        if (!nextFiles) {
          if (imageAttachment.url) URL.revokeObjectURL(imageAttachment.url)
          setUploadError(FILE_TOO_LARGE_MESSAGE)
          continue
        }
        attachedFilesRef.current = nextFiles
        setAttachedFiles(nextFiles)
      } catch (error) {
        if (generation !== uploadGenerationRef.current) return
        console.error("Error processing file:", error)
        setUploadError("Failed to process file")
      }
    }
  }, [disableImageUploads])

  /**
   * Handle file selection from input element.
   */
  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    const selectedFiles = Array.from(files)
    // Allow selecting the same file again after removal, even during conversion.
    event.target.value = ""
    await processFiles(selectedFiles)
  }, [processFiles])

  /**
   * Handle paste events (images from clipboard).
   */
  const handlePaste = useCallback(async (event: React.ClipboardEvent) => {
    const items = event.clipboardData?.items
    if (!items) return

    const files = Array.from(items)
      .filter(item => item.type.startsWith("image/"))
      .map(item => item.getAsFile())
      .filter((file): file is File => file !== null)
    if (files.length === 0) return // Ordinary text paste belongs to the editor.
    event.preventDefault()
    await processFiles(files)
  }, [processFiles])

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
    const file = attachedFilesRef.current.find(file => file.id === fileId)
    if (file?.url) URL.revokeObjectURL(file.url)
    attachedFilesRef.current = attachedFilesRef.current.filter(file => file.id !== fileId)
    setAttachedFiles(attachedFilesRef.current)
  }, [])

  /**
   * Clear all attached files.
   * Note: We don't revoke URLs here because the message that was just sent
   * still needs them for rendering. URLs will be cleaned up by the browser
   * when the page is closed or refreshed.
   */
  const clearFiles = useCallback(() => {
    // A file still converting at send time belongs to the old draft.
    uploadGenerationRef.current += 1
    attachedFilesRef.current = []
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
