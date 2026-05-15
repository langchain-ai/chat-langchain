/**
 * Welcome Screen Component
 *
 * Displays a centered welcome screen for new chats with the LangChain logo,
 * animated text, and a centered input box.
 */

"use client"

import React from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { FilePreviewGrid } from "./file-preview-grid"
import { VoiceInputButton } from "./voice-input-button"
import type { ImageAttachment } from "@/lib/types"
import type { AgentConfig } from "@/components/layout/agent-settings"
import { MAX_INPUT_CHARS } from "@/lib/constants/features"
import {
  getAllowedModels,
  getModelDisplayName,
  type ModelOption,
} from "@/lib/config/deployment-config"

interface WelcomeScreenProps {
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

  // Agent configuration
  agentConfig?: AgentConfig
  onAgentConfigChange?: (config: AgentConfig) => void
}

/**
 * Welcome screen shown when starting a new chat.
 * Features a centered input box with file upload support.
 */
export function WelcomeScreen({
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
  agentConfig,
  onAgentConfigChange,
}: WelcomeScreenProps) {
  const allowedModels = getAllowedModels()

  const handleModelChange = (model: string) => {
    if (agentConfig && onAgentConfigChange) {
      onAgentConfigChange({ ...agentConfig, model })
    }
  }

  return (
    <div className="absolute inset-0 flex items-center justify-center px-3 sm:px-4">
      <div className="w-full max-w-3xl -mt-10 sm:-mt-20">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="mb-6 flex justify-center">
            <Image
              src="/assets/images/LangChain_Symbol_LightBlue.svg"
              alt="LangChain"
              width={68}
              height={68}
              priority
            />
          </div>
          <h2 className="text-2xl sm:text-4xl font-semibold text-white mb-2" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
            What can I help with?
          </h2>
        </div>

        {/* File Previews */}
        <FilePreviewGrid files={attachedFiles} onRemove={onRemoveFile} />

        {/* Upload Error */}
        {uploadError && (
          <div className="mb-3 text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-md">
            {uploadError}
          </div>
        )}

        {/* Voice Error */}
        {voiceError && (
          <div className="mb-3 text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-md">
            {voiceError}
          </div>
        )}

        {/* Centered Input Container */}
        <div className="relative group">
          <div
            className={`relative bg-card border-2 rounded-2xl shadow-2xl transition-all duration-300 border-primary/60 ring-1 ring-primary/20 ${
              isDragging
                ? 'border-primary bg-primary/5 ring-2 ring-primary/30'
                : 'group-hover:border-primary/80 group-focus-within:border-primary/90 group-focus-within:ring-2 group-focus-within:ring-primary/30'
            }`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            {isDragging && (
              <div className="absolute inset-0 bg-primary/10 rounded-2xl flex items-center justify-center z-20 pointer-events-none">
                <div className="text-primary font-medium">Drop files here</div>
              </div>
            )}
            <div className="flex items-end gap-2 px-4 py-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.py,.js,.ts,.tsx,.jsx,.java,.cpp,.c,.h,.cs,.go,.rs,.rb,.php,.sh,.bash,.yaml,.yml,.json,.xml,.html,.css,.md,.txt,.log,.sql,.graphql,.r,.swift,.kt,.scala,.har"
                multiple
                onChange={onFileSelect}
                className="hidden"
              />

              {!isLoading && (
                <Button
                  onClick={onFileButtonClick}
                  variant="ghost"
                  size="sm"
                  disabled={isLoading || !userId}
                  className="group h-10 w-10 p-0 mb-0.5 rounded-full bg-muted/50 hover:bg-primary/10 text-muted-foreground hover:text-primary border-0 flex-shrink-0 transition-all duration-200 hover:scale-105 active:scale-95"
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
                    className="w-5 h-5"
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
                placeholder={userId ? "Ask me anything about LangChain..." : "Initializing..."}
                className="relative z-10 min-h-[48px] max-h-[240px] resize-none bg-transparent border-0 w-full px-3 py-3 text-base leading-relaxed text-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 transition-all duration-200 break-words custom-scrollbar"
                disabled={isLoading || !userId}
                rows={1}
              />

              {isVoiceSupported && onVoiceToggle && (
                <VoiceInputButton
                  isListening={isVoiceListening ?? false}
                  disabled={!userId}
                  onClick={onVoiceToggle}
                  size="md"
                />
              )}

              {isLoading && (
                <Button
                  onClick={onStop}
                  variant="ghost"
                  size="sm"
                  disabled={isStopping}
                  className={`
                    h-10 px-5 mb-0.5 rounded-full flex-shrink-0
                    transition-all duration-200 hover:scale-105 active:scale-95
                    bg-muted text-primary hover:text-primary hover:bg-muted/80 border-2 border-primary
                    ${isStopping ? 'opacity-60 cursor-not-allowed' : ''}
                  `}
                  type="button"
                  title={isStopping ? "Stopping..." : "Stop generating"}
                >
                  <span className="text-sm font-medium">
                    {isStopping ? 'Stopping...' : 'Stop'}
                  </span>
                </Button>
              )}
            </div>
          </div>

          {inputError && (
            <div className="mt-2 px-2 text-sm text-destructive">
              {inputError}
            </div>
          )}

          {/* Model selector dropdown - positioned underneath chatbox in bottom left */}
          {agentConfig && onAgentConfigChange && (
            <div className="flex justify-start mt-2 px-2">
              <Select value={agentConfig.model} onValueChange={handleModelChange}>
                <SelectTrigger className="h-8 text-sm border-0 bg-transparent hover:bg-muted/50 px-2 gap-1 w-auto">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {allowedModels.map((model) => (
                    <SelectItem key={model} value={model}>
                      {getModelDisplayName(model as ModelOption)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
