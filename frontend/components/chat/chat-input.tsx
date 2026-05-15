/**
 * Chat Input Component
 *
 * Fixed input area at the bottom of the chat interface.
 * Includes file upload, drag & drop, and paste support.
 */

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { FilePreviewGrid } from "./features/file-preview-grid"
import { VoiceInputButton } from "./features/voice-input-button"
import type { ImageAttachment } from "@/lib/types"
import { MAX_INPUT_CHARS } from "@/lib/constants/features"

interface ChatInputProps {
  input: string
  onInputChange: (value: string) => void
  onBeforeInput: (e: React.FormEvent<HTMLTextAreaElement>) => void
  onSend: () => void
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  isLoading: boolean
  isStopping: boolean
  onStop: () => void
  userId?: string | null

  // File upload
  attachedFiles: ImageAttachment[]
  uploadError: string | null
  inputError: string | null
  isDragging: boolean
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  onPaste: (e: React.ClipboardEvent<HTMLTextAreaElement>) => void
  onRemoveFile: (fileId: string) => void
  onFileButtonClick: (e: React.MouseEvent) => void
  fileInputRef: React.RefObject<HTMLInputElement>
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
  textareaRef?: React.RefObject<HTMLTextAreaElement>

  // Voice input
  isVoiceListening?: boolean
  isVoiceSupported?: boolean
  onVoiceToggle?: () => void
  voiceError?: string | null

  // Queued messages
  queuedMessages?: { content: string; id: string }[]
}

/**
 * Chat input area with file upload support.
 * Displays at the bottom of the chat interface when there are existing messages.
 */
export function ChatInput({
  input,
  onInputChange,
  onBeforeInput,
  onSend,
  onKeyDown,
  isLoading,
  isStopping,
  onStop,
  userId,
  attachedFiles,
  uploadError,
  inputError,
  isDragging,
  onDragOver,
  onDragLeave,
  onDrop,
  onPaste,
  onRemoveFile,
  onFileButtonClick,
  fileInputRef,
  onFileSelect,
  textareaRef,
  isVoiceListening,
  isVoiceSupported,
  onVoiceToggle,
  voiceError,
  queuedMessages = [],
}: ChatInputProps) {
  return (
    <div className="relative">
      {/* Enhanced visibility layer */}
      <div className="absolute inset-0 bg-card/20 pointer-events-none" />

      <div className="relative border-t border-border/60 bg-background backdrop-blur-sm">
        <div className="w-full max-w-4xl mx-auto px-3 sm:px-4 py-1.5">
          {/* File Previews */}
          <FilePreviewGrid files={attachedFiles} onRemove={onRemoveFile} />

          {/* Upload Error */}
          {uploadError && (
            <div className="mb-2 text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-md">
              {uploadError}
            </div>
          )}

          {/* Voice Error */}
          {voiceError && (
            <div className="mb-2 text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-md">
              {voiceError}
            </div>
          )}

          {/* Queued Messages */}
          {queuedMessages.length > 0 && (
            <div className="mb-2 space-y-1.5">
              {queuedMessages.map((msg) => (
                <div
                  key={msg.id}
                  className="flex items-center gap-2 px-3 py-2 bg-muted/50 border border-border/50 rounded-lg text-sm"
                >
                  <div className="flex items-center gap-1.5 text-muted-foreground flex-shrink-0">
                    <svg
                      className="w-3 h-3 animate-pulse"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                    >
                      <circle cx="12" cy="12" r="10" />
                    </svg>
                    <span className="text-xs font-medium">Queued</span>
                  </div>
                  <span className="text-foreground/80 truncate">{msg.content}</span>
                </div>
              ))}
            </div>
          )}

          <div className="relative group">
            {/* Multi-layered input container */}
            <div className="relative">
              {/* High-contrast glow layer for visibility */}
              <div className="absolute -inset-1 bg-primary/8 rounded-2xl opacity-70 group-hover:opacity-90 group-focus-within:opacity-100 transition-opacity duration-300 shadow-2xl" />

              {/* Main input container with enhanced contrast */}
              <div
                className={`relative bg-card/95 backdrop-blur-sm border-2 rounded-xl shadow-2xl transition-all duration-300 group-hover:shadow-3xl group-hover:bg-card/98 group-focus-within:shadow-3xl group-focus-within:bg-white/5 group-focus-within:ring-2 group-focus-within:ring-primary/20 ${
                  isDragging
                    ? 'border-primary bg-primary/5 ring-2 ring-primary/30'
                    : 'border-border/50 group-hover:border-primary/60 group-focus-within:border-primary/70'
                }`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
              >
                {isDragging && (
                  <div className="absolute inset-0 bg-primary/10 rounded-xl flex items-center justify-center z-20 pointer-events-none">
                    <div className="text-primary font-medium">Drop files here</div>
                  </div>
                )}
                <div className="flex items-end gap-2 px-3 py-1.5">
                  {/* Hidden File Input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,.py,.js,.ts,.tsx,.jsx,.java,.cpp,.c,.h,.cs,.go,.rs,.rb,.php,.sh,.bash,.yaml,.yml,.json,.xml,.html,.css,.md,.txt,.log,.sql,.graphql,.r,.swift,.kt,.scala,.har"
                    multiple
                    onChange={onFileSelect}
                    className="hidden"
                  />

                  {/* File Upload Button - Stays at bottom as textarea grows */}
                  {!isLoading && (
                    <Button
                      onClick={onFileButtonClick}
                      variant="ghost"
                      size="sm"
                      disabled={isLoading || !userId}
                      className="group h-9 w-9 p-0 mb-0.5 rounded-full bg-muted/50 hover:bg-primary/10 text-muted-foreground hover:text-primary border-0 flex-shrink-0 transition-all duration-200 hover:scale-105 active:scale-95"
                      type="button"
                      title="Attach files (images, code, logs)"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="w-4.5 h-4.5"
                      >
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                      </svg>
                    </Button>
                  )}

                  <Textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => onInputChange(e.target.value)}
                    onBeforeInput={onBeforeInput}
                    onKeyDown={onKeyDown}
                    onPaste={onPaste}
                    maxLength={MAX_INPUT_CHARS}
                    placeholder={
                      !userId
                        ? "Initializing..."
                        : isLoading
                          ? "Type your next message..."
                          : "Ask me anything about LangChain..."
                    }
                    className="relative z-10 min-h-[36px] max-h-[240px] resize-none bg-transparent border-0 w-full px-3 py-2 text-sm leading-relaxed text-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 transition-all duration-200 break-words custom-scrollbar"
                    disabled={!userId}
                    rows={1}
                  />

                  {isVoiceSupported && onVoiceToggle && (
                    <VoiceInputButton
                      isListening={isVoiceListening ?? false}
                      disabled={!userId}
                      onClick={onVoiceToggle}
                      size="sm"
                    />
                  )}

                  {isLoading && (
                    <Button
                      onClick={onStop}
                      variant="ghost"
                      size="sm"
                      disabled={isStopping}
                      className={`
                        h-9 px-4 mb-0.5 rounded-full flex-shrink-0
                        transition-all duration-200 hover:scale-105 active:scale-95
                        bg-muted text-primary hover:text-primary hover:bg-muted/80 border-2 border-primary
                        ${isStopping ? 'opacity-60 cursor-not-allowed' : ''}
                      `}
                      type="button"
                      title={isStopping ? "Stopping..." : "Stop generating"}
                    >
                      <span className="text-xs font-medium">
                        {isStopping ? 'Stopping...' : 'Stop'}
                      </span>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </div>

          {inputError && (
            <div className="mt-1 px-2 text-xs text-destructive">
              {inputError}
            </div>
          )}

          {/* Simple help text - hidden on mobile */}
          <div className="hidden sm:flex items-center justify-between mt-1 px-2">
            <p className="text-[11px] text-muted-foreground/60">
              <kbd className="px-1 py-0.5 bg-muted/50 rounded text-[10px] font-medium text-foreground/70">Enter</kbd> to send
              <span className="mx-1">•</span>
              <kbd className="px-1 py-0.5 bg-muted/50 rounded text-[10px] font-medium text-foreground/70">Shift+Enter</kbd> new line
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
