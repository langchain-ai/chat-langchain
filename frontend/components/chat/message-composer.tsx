"use client"

import { useId, useRef } from "react"
import {
  ChatComposer,
  ChatComposerAction,
  ChatComposerToolbar,
  type ChatComposerInputHandle,
} from "@langchain/macaw-components/ChatComposer"
import { ChatComposerPlainInput } from "@langchain/macaw-components/ChatComposer/ChatComposerPlainInput"
import { Button } from "@langchain/macaw-components/Button"
import { ErrorMessage } from "@langchain/macaw-components/ErrorMessage"
import { IconButton } from "@langchain/macaw-components/IconButton"
import { Kbd } from "@langchain/macaw-components/Kbd"
import { Text } from "@langchain/macaw-components/Text"
import { PlusIcon } from "@langchain/macaw-components/icons"
import { FilePreviewGrid } from "./features/file-preview-grid"
import { VoiceInputButton } from "./features/voice-input-button"
import type { ImageAttachment } from "@/lib/types"

export interface MessageComposerProps {
  input: string
  onInputChange: (value: string) => void
  onSend: () => void
  isLoading: boolean
  isStopping: boolean
  onStop: () => void
  userId?: string | null
  attachedFiles: ImageAttachment[]
  uploadError: string | null
  inputError: string | null
  isDragging: boolean
  onDragOver: (event: React.DragEvent) => void
  onDragLeave: (event: React.DragEvent) => void
  onDrop: (event: React.DragEvent) => void
  onPaste: (event: React.ClipboardEvent) => void
  onRemoveFile: (fileId: string) => void
  onFileButtonClick: (event: React.MouseEvent) => void
  fileInputRef: React.RefObject<HTMLInputElement | null>
  onFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => void
  textareaRef?: React.RefObject<ChatComposerInputHandle | null>
  isVoiceListening?: boolean
  isVoiceSupported?: boolean
  onVoiceToggle?: () => void
  voiceError?: string | null
  queuedMessages?: { content: string; id: string }[]
}

const acceptedFiles = "image/*,.py,.js,.ts,.tsx,.jsx,.java,.cpp,.c,.h,.cs,.go,.rs,.rb,.php,.sh,.bash,.yaml,.yml,.json,.xml,.html,.css,.md,.txt,.log,.sql,.graphql,.r,.swift,.kt,.scala,.har"

/** Translate app-owned draft, upload, speech, and queue state into Macaw slots. */
export function MessageComposer({
  input,
  onInputChange,
  onSend,
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
  isVoiceListening = false,
  isVoiceSupported,
  onVoiceToggle,
  voiceError,
  queuedMessages = [],
}: MessageComposerProps) {
  const localInputRef = useRef<ChatComposerInputHandle>(null)
  const inputRef = textareaRef ?? localInputRef
  const helpId = useId()
  const errorId = useId()
  const errors = [uploadError, voiceError, inputError].filter(Boolean)
  const disabled = !userId
  const canSend = !disabled && (input.trim().length > 0 || attachedFiles.length > 0)

  return (
    <div className="space-y-2">
      <ChatComposer
        aria-label="Message composer"
        disabled={disabled}
        submitDisabled={!canSend}
        onSubmit={onSend}
        onDragOver={(event) => {
          if (disabled) {
            event.preventDefault()
            return
          }
          onDragOver(event)
        }}
        onDragLeave={onDragLeave}
        onDrop={(event) => {
          if (disabled) {
            event.preventDefault()
            return
          }
          onDrop(event)
        }}
        onPasteCapture={(event) => {
          if (!disabled) onPaste(event)
        }}
        onClick={(event) => {
          // Focus only blank composer space; leave editor selections and controls alone.
          const target = event.target as HTMLElement
          if (!disabled && !target.closest("button, input, textarea, a, [contenteditable], [role='textbox']")) {
            inputRef.current?.focus()
          }
        }}
        drawer={queuedMessages.length > 0 ? (
          <div className="space-y-1 p-3" aria-live="polite" aria-label="Queued messages">
            {queuedMessages.map((message) => (
              <div key={message.id} className="flex min-w-0 items-center gap-2">
                <Text as="span" variant="xs" color="secondary">Queued</Text>
                <Text as="span" variant="sm" className="truncate">{message.content}</Text>
              </div>
            ))}
          </div>
        ) : undefined}
      >
        {isDragging && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-xl border border-focus bg-surface-level-2/95" role="status">
            <Text weight="medium">Drop files here</Text>
          </div>
        )}
        <FilePreviewGrid files={attachedFiles} onRemove={onRemoveFile} />
        {/* Chat prompts are raw text: rich Markdown serialization changes code and indentation. */}
        <ChatComposerPlainInput
          ref={inputRef}
          value={input}
          onChange={onInputChange}
          placeholder={disabled ? "Initializing…" : isLoading ? "Type your next message…" : "Ask about LangChain…"}
          aria-label="Message"
          aria-describedby={`${helpId}${errors.length ? ` ${errorId}` : ""}`}
          minHeight={64}
          maxHeight={240}
        />
        <ChatComposerToolbar
          actions={(
            <>
              {isLoading && (
                <Button
                  type="button"
                  size="sm"
                  variant="outlined"
                  onClick={onStop}
                  disabled={isStopping}
                  aria-label={isStopping ? "Stopping response" : "Stop response"}
                >
                  {isStopping ? "Stopping…" : "Stop"}
                </Button>
              )}
              <ChatComposerAction label={isLoading ? "Queue message" : "Send message"} />
            </>
          )}
        >
          <input
            ref={fileInputRef}
            type="file"
            aria-label="Upload files"
            accept={acceptedFiles}
            multiple
            onChange={onFileSelect}
            disabled={disabled || isLoading}
            className="hidden"
          />
          <IconButton
            type="button"
            icon={PlusIcon}
            label="Attach files (images, code, logs)"
            variant="plain"
            size="sm"
            onClick={onFileButtonClick}
            disabled={disabled || isLoading}
          />
          {isVoiceSupported && onVoiceToggle && (
            <VoiceInputButton
              isListening={isVoiceListening}
              onClick={onVoiceToggle}
              disabled={disabled}
            />
          )}
        </ChatComposerToolbar>
      </ChatComposer>
      {errors.length > 0 && (
        <div id={errorId} role="alert" className="space-y-1">
          {errors.map((error, index) => <ErrorMessage key={index} message={error} size="sm" />)}
        </div>
      )}
      {/* Key hints only apply to hardware keyboards; the voice status still shows on touch screens. */}
      <div id={helpId} className={`flex flex-wrap items-center justify-between gap-2 px-1${isVoiceListening ? "" : " max-sm:hidden"}`}>
        <Text variant="xs" color="tertiary" className="max-sm:hidden">
          <Kbd>Enter</Kbd> {isLoading ? "to queue" : "to send"}
          <span className="mx-2">·</span>
          <Kbd>Shift Enter</Kbd> for a new line
        </Text>
        {isVoiceListening && <Text variant="xs" color="secondary" role="status">Listening…</Text>}
      </div>
    </div>
  )
}
