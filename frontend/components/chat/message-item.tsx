import { Copy, Check, Settings, RefreshCw, ThumbsUp, ThumbsDown, MessageSquare, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { ThinkingTimer } from "./animations/thinking-timer"
import { AnimatedThinking } from "./animations/animated-thinking"
import type { Message } from "@/lib/types"
import { useState, useMemo, useEffect, useCallback, memo, useRef } from "react"
import Image from "next/image"

// ============================================================================
// Constants
// ============================================================================

const COPY_FEEDBACK_DURATION = 2000

// Color palette for code highlighting
const CODE_COLORS = {
  // Background & borders
  blockBackground: 'oklch(0.16 0 0)',
  blockBorder: 'oklch(0.30 0 0)',
  inlineBackground: 'oklch(0.22 0 0)',
  inlineBorder: 'oklch(0.32 0 0)',

  // Primary theme colors
  primary: '#7FC8FF',      // Blue - used for properties, operators, tags
  primaryLight: '#99D3FF',  // Light blue - strings, attributes
  primaryDark: '#B2DEFF',  // Lighter blue - keywords, built-ins

  // Accent colors
  blue: '#60a5fa',         // Functions
  yellow: '#fbbf24',       // Classes
  orange: '#f59e0b',       // Numbers, booleans
  green: '#10b981',        // Selectors, inserted
  red: '#ef4444',          // Important, deleted

  // Neutral colors
  text: '#e4e4e7',         // Main text
  comment: '#6b7280',      // Comments, docstrings
  punctuation: '#a1a1aa',  // Punctuation
} as const

// ============================================================================
// Syntax Highlighting Theme
// ============================================================================

const customTheme = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: CODE_COLORS.blockBackground,
    border: `1px solid ${CODE_COLORS.blockBorder}`,
    borderRadius: '8px',
    padding: '1rem',
    margin: '0.75rem 0',
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    background: 'transparent',
    color: CODE_COLORS.text,
    fontSize: '13px',
    lineHeight: '1.6',
  },
  // Token colors - grouped by theme color
  'comment': { color: CODE_COLORS.comment },
  'prolog': { color: CODE_COLORS.comment },
  'doctype': { color: CODE_COLORS.comment },
  'cdata': { color: CODE_COLORS.comment },
  'punctuation': { color: CODE_COLORS.punctuation },

  'property': { color: CODE_COLORS.primary },
  'tag': { color: CODE_COLORS.primary },
  'operator': { color: CODE_COLORS.primary },
  'entity': { color: CODE_COLORS.primary },
  'url': { color: CODE_COLORS.primary },
  'attr-name': { color: CODE_COLORS.primary },

  'string': { color: CODE_COLORS.primaryLight },
  'char': { color: CODE_COLORS.primaryLight },
  'attr-value': { color: CODE_COLORS.primaryLight },

  'builtin': { color: CODE_COLORS.primaryDark },
  'atrule': { color: CODE_COLORS.primaryDark },
  'keyword': { color: CODE_COLORS.primaryDark },

  'boolean': { color: CODE_COLORS.orange },
  'number': { color: CODE_COLORS.orange },
  'constant': { color: CODE_COLORS.orange },
  'symbol': { color: CODE_COLORS.orange },
  'regex': { color: CODE_COLORS.orange },

  'selector': { color: CODE_COLORS.green },
  'inserted': { color: CODE_COLORS.green },

  'function': { color: CODE_COLORS.blue },
  'class-name': { color: CODE_COLORS.yellow },
  'variable': { color: CODE_COLORS.text },

  'important': { color: CODE_COLORS.red, fontWeight: 'bold' },
  'deleted': { color: CODE_COLORS.red },
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Recursively extract text content from ReactMarkdown nodes
 * Used for extracting code from markdown code blocks for copy functionality
 */
const extractTextFromNode = (node: any): string => {
  if (typeof node === 'string') return node
  if (node?.props?.children) {
    if (typeof node.props.children === 'string') {
      return node.props.children
    }
    if (Array.isArray(node.props.children)) {
      return node.props.children.map(extractTextFromNode).join('')
    }
    return extractTextFromNode(node.props.children)
  }
  return ''
}

/**
 * Individual code block component with its own copy state
 * This prevents the copy button from flickering during streaming
 */
const CodeBlock = memo(({ codeString, language }: { codeString: string; language: string }) => {
  const [isCopied, setIsCopied] = useState(false)

  const handleCopyCode = useCallback(() => {
    navigator.clipboard.writeText(codeString)
    setIsCopied(true)
    setTimeout(() => setIsCopied(false), COPY_FEEDBACK_DURATION)
  }, [codeString])

  return (
    <div className="relative group my-4">
      <SyntaxHighlighter
        language={language}
        style={customTheme}
        customStyle={{
          margin: '0.75rem 0',
          background: CODE_COLORS.blockBackground,
          border: `1px solid ${CODE_COLORS.blockBorder}`,
          borderRadius: '8px',
          padding: '1rem',
        }}
        codeTagProps={{
          style: {
            fontSize: '13px',
            fontFamily: 'var(--font-mono), ui-monospace, monospace',
          }
        }}
      >
        {codeString}
      </SyntaxHighlighter>
      <button
        onClick={handleCopyCode}
        className="absolute top-2 right-2 sm:top-3 sm:right-3 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity duration-200 px-2 sm:px-2.5 py-1 sm:py-1.5 rounded-md text-xs flex items-center gap-1 sm:gap-1.5 backdrop-blur-sm"
        style={{
          background: 'rgba(0, 0, 0, 0.7)',
          color: CODE_COLORS.text,
          border: `1px solid ${CODE_COLORS.blockBorder}`,
          willChange: 'opacity',
        }}
        aria-label="Copy code to clipboard"
      >
        {isCopied ? (
          <>
            <Check className="w-3.5 h-3.5" />
            Copied
          </>
        ) : (
          <>
            <Copy className="w-3.5 h-3.5" />
            Copy
          </>
        )}
      </button>
    </div>
  )
})

interface MessageItemProps {
  message: Message
  showToolCalls?: boolean
  isLastAssistant: boolean
  isRegenerating: boolean
  copiedId: string | null
  onCopy: (content: string, messageId: string) => void
  onRegenerate: () => void
  onEditAndRerun?: (messageId: string, newContent: string) => void
  feedbackComment: { [messageId: string]: string }
  showCommentInput: string | null
  onFeedback: (messageId: string, feedbackType: "positive" | "negative", comment?: string) => void
  onSubmitComment: (messageId: string) => void
  onCancelComment: (messageId: string) => void
  onToggleComment: (messageId: string) => void
  setFeedbackComment: React.Dispatch<React.SetStateAction<{ [messageId: string]: string }>>
}

export const MessageItem = memo(function MessageItem({
  message,
  showToolCalls,
  isLastAssistant,
  isRegenerating,
  copiedId,
  onCopy,
  onRegenerate,
  onEditAndRerun,
  feedbackComment,
  showCommentInput,
  onFeedback,
  onSubmitComment,
  onCancelComment,
  onToggleComment,
  setFeedbackComment,
}: MessageItemProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState(message.content)
  const prevContentRef = useRef(message.content)

  // Sync editContent when message.content changes (e.g., during streaming)
  useEffect(() => {
    if (!isEditing && message.content !== prevContentRef.current) {
      setEditContent(message.content)
      prevContentRef.current = message.content
    }
  }, [message.content, isEditing])

  const handleSaveEdit = useCallback(() => {
    if (editContent.trim() && onEditAndRerun) {
      onEditAndRerun(message.id, editContent.trim())
      setIsEditing(false)
    }
  }, [editContent, onEditAndRerun, message.id])

  const handleCancelEdit = useCallback(() => {
    setEditContent(message.content)
    setIsEditing(false)
  }, [message.content])

  // Track code block index to generate stable IDs during streaming
  const codeBlockIndexRef = useRef(0)

  // Reset counter before each render so code blocks get consistent indices
  codeBlockIndexRef.current = 0

  // Memoize markdown components to prevent button remounting during streaming
  const markdownComponents = useMemo(() => ({
    // Custom link renderer - opens in new tab
    a: ({ children, ...props }: any) => (
      <a
        {...props}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          color: CODE_COLORS.primary,
          textDecorationColor: CODE_COLORS.primary
        }}
      >
        {children}
      </a>
    ),

    // Custom code renderer - handles both inline code and code blocks
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '')
      const language = match ? match[1] : 'text'
      const codeString = String(children).replace(/\n$/, '')

      // Check if it's inline code: single backticks or no newlines
      const isInlineCode = inline === true || (!className && !codeString.includes('\n'))

      // Inline code (single backticks) - Slack-style highlighting
      if (isInlineCode) {
        return (
          <code
            className="px-1.5 py-0.5 text-[13px] font-mono"
            style={{
              backgroundColor: CODE_COLORS.inlineBackground,
              color: CODE_COLORS.primary,
              border: `1px solid ${CODE_COLORS.inlineBorder}`,
              borderRadius: '5px',
            }}
            {...props}
          >
            {children}
          </code>
        )
      }

      // Code blocks (triple backticks) - use stable ID based on position, not content
      // This prevents flickering during streaming when code content changes
      const blockIndex = codeBlockIndexRef.current++
      const codeBlockId = `${message.id}-code-${blockIndex}`

      // Render a separate component for the code block with copy functionality
      return <CodeBlock key={codeBlockId} codeString={codeString} language={language} />
    },
  }), [message.id])

  return (
    <>
      <style jsx>{`
        @keyframes dance {
          0% { transform: rotate(-30deg) scale(1); }
          25% { transform: rotate(0deg) scale(1.05); }
          50% { transform: rotate(30deg) scale(1); }
          75% { transform: rotate(0deg) scale(1.05); }
          100% { transform: rotate(-30deg) scale(1); }
        }
        @keyframes spin360 {
          0%, 90% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(5px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .dance-wrapper {
          animation: spin360 6s linear infinite;
        }
        .dancing {
          animation: dance 0.8s ease-in-out infinite;
        }

        /* Smooth text rendering optimizations */
        .prose {
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          text-rendering: optimizeLegibility;
        }

        /* Optimize layout performance during streaming */
        .prose > * {
          transition: opacity 0.1s ease-out;
        }
      `}</style>
      <div className="flex gap-3 sm:gap-4 items-start group/message">
        <div
          className={`w-8 h-8 flex items-center justify-center flex-shrink-0 ${
            message.role === "assistant" && message.isThinking ? "dance-wrapper" : ""
          }`}
        >
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center"
          >
            {message.role === "assistant" ? (
              <Image
                src="/assets/images/Assistant Icon.png"
                alt="Assistant Logo"
                width={32}
                height={32}
                className="object-contain"
              />
            ) : (
              <Image
                src="/assets/images/User icon.png"
                alt="User"
                width={32}
                height={32}
                className="object-contain"
              />
            )}
          </div>
        </div>
      <div className="flex-1 min-w-0 space-y-2">
        <div
          className={`rounded-lg px-4 py-3 transition-all duration-150 ease-out ${
            message.role === "user" ? "bg-muted/50 text-foreground" : "bg-muted text-foreground"
          }`}
          style={{
            willChange: message.isThinking ? 'contents' : 'auto',
            contain: 'layout style paint',
          }}
        >
          {/* Thinking indicator - only for assistant messages */}
          {message.role === "assistant" && (message.isThinking || message.thinkingStartTime || (message.thinkingSteps && message.thinkingSteps.length > 0)) && (
            <details open className="mb-3 text-xs">
              <summary className="cursor-pointer flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
                {message.isThinking && <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />}
                <span>
                  {message.isThinking ? <AnimatedThinking /> : <span className="font-medium">Agent steps</span>} ({message.thinkingSteps?.length || 0})
                </span>
                <span className="ml-1">•</span>
                <ThinkingTimer
                  startTime={message.thinkingStartTime}
                  duration={message.thinkingDuration}
                  isThinking={!!message.isThinking}
                />
              </summary>
              {message.thinkingSteps && message.thinkingSteps.length > 0 && (
                <div className="mt-2 pl-4 space-y-1 text-muted-foreground font-mono text-[11px]">
                  {message.thinkingSteps.map((step, idx) => (
                    <div key={`${message.id}-step-${idx}`} className="flex items-start gap-2">
                      <span className="text-primary opacity-50">{(idx + 1).toString().padStart(2, '0')}</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              )}
            </details>
          )}

          {message.role === "user" ? (
            isEditing ? (
              <Textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="min-h-[80px] text-sm"
                autoFocus
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleSaveEdit()
                    } else if (e.key === "Escape") {
                      handleCancelEdit()
                    }
                  }}
                  onBlur={handleCancelEdit}
                  onFocus={(e) => {
                    // Select all text on focus for easier editing
                    e.target.select()
                  }}
              />
            ) : (
              <div className="space-y-2">
                {/* File attachments - uniform grid layout */}
                {message.images && message.images.length > 0 && (
                  <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-2 mb-3">
                    {message.images.map((file) => {
                      const isImage = file.mimeType?.startsWith('image/')
                      const fileName = file.name || "File"
                      const fileExt = fileName.split('.').pop()?.toLowerCase()
                      const fileSizeKB = file.size ? Math.round(file.size / 1024) : 0

                      // Get file type icon color
                      const getFileColor = () => {
                        return "text-white"
                      }

                      return (
                        <div
                          key={file.id}
                          className="h-32 rounded-lg border-2 border-border bg-muted/30 hover:bg-muted/50 hover:border-primary transition-all flex flex-col overflow-hidden"
                        >
                          {isImage ? (
                            // Image with filename overlay
                            <div className="relative h-full w-full">
                              <img
                                src={file.url || `data:${file.mimeType};base64,${file.base64}`}
                                alt={fileName}
                                className="h-full w-full object-cover"
                              />
                              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1">
                                <p className="text-xs text-white truncate" title={fileName}>
                                  {fileName}
                                </p>
                              </div>
                            </div>
                          ) : (
                            // File card with icon
                            <div className="h-full flex flex-col items-center justify-center p-3 text-center">
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className={`w-10 h-10 mb-2 ${getFileColor()}`}
                              >
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                              </svg>
                              <span className="text-xs font-medium text-foreground truncate w-full px-1 mb-1" title={fileName}>
                                {fileName}
                              </span>
                              <div className="flex items-center gap-1.5">
                                <span className={`text-xs font-bold px-1.5 py-0.5 rounded bg-muted ${getFileColor()}`}>
                                  {fileExt?.toUpperCase().slice(0, 4)}
                                </span>
                                {fileSizeKB > 0 && (
                                  <span className="text-xs text-muted-foreground">
                                    {fileSizeKB}KB
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
                {/* Text content */}
                {message.content && (
                  <p
                    className="text-sm leading-relaxed whitespace-pre-wrap break-words cursor-pointer rounded px-2 py-1 -mx-2 -my-1 transition-colors overflow-wrap break-word"
                    onClick={() => onEditAndRerun && setIsEditing(true)}
                    title="Click to edit and rerun from here"
                  >
                    {String(message.content || "")}
                  </p>
                )}
              </div>
            )
          ) : (
            <div className="relative">
              <div
                className="text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none break-words overflow-wrap break-word transition-opacity duration-200 ease-out"
                style={{
                  animation: message.isThinking ? 'none' : 'fadeIn 0.3s ease-out',
                  willChange: message.isThinking ? 'contents, opacity' : 'auto',
                  backfaceVisibility: 'hidden',
                  transform: 'translateZ(0)',
                }}
              >
                {message.content && typeof message.content === 'string' ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {message.content}
                  </ReactMarkdown>
              ) : null}
              </div>

              {/* Metadata in bottom right of message box */}
              {!message.isThinking && message.runId && (message.shareUrl || message.usageMetadata?.total_tokens || message.thinkingDuration) && (
                <div className="flex items-center justify-end gap-2 mt-3 text-xs text-white font-mono font-medium">
                  <>
                    {message.thinkingDuration && (
                      <span>{(message.thinkingDuration / 1000).toFixed(1)}s</span>
                    )}
                    {message.thinkingDuration && (message.usageMetadata?.total_tokens || message.shareUrl) && (
                      <span>•</span>
                    )}
                    {message.usageMetadata?.total_tokens && (
                      <span>{(message.usageMetadata.total_tokens / 1000).toFixed(1)}k tokens</span>
                    )}
                    {message.usageMetadata?.total_tokens &&
                      typeof message.usageMetadata.total_cost === "number" &&
                      message.usageMetadata.total_cost > 0 && (
                      <>
                        <span>•</span>
                        <span>${message.usageMetadata.total_cost.toFixed(4)}</span>
                      </>
                      )}
                    {message.usageMetadata?.total_tokens && message.shareUrl && <span>•</span>}
                    {message.shareUrl && (
                      <a
                        href={message.shareUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-0.5 text-xs font-medium transition-colors ml-1"
                        style={{ color: '#7FC8FF', textDecorationColor: '#7FC8FF' }}
                      >
                        View trace
                        <ExternalLink className="h-2.5 w-2.5" />
                      </a>
                    )}
                  </>
                </div>
              )}

            </div>
          )}
        </div>

        {message.role === "assistant" && (
          <>
            <div className="flex gap-1 sm:gap-2 items-center flex-wrap">
              {!message.isThinking && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onCopy(message.content, message.id)}
                    className="h-8 px-2 text-xs"
                  >
                    {copiedId === message.id ? (
                      <>
                        <Check className="w-3 h-3 mr-1" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3 mr-1" />
                        Copy
                      </>
                    )}
                  </Button>

                  {isLastAssistant && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onRegenerate}
                      disabled={isRegenerating}
                      className="h-8 px-2 text-xs"
                    >
                      <RefreshCw className={`w-3 h-3 mr-1 ${isRegenerating ? "animate-spin" : ""}`} />
                      Regenerate
                    </Button>
                  )}
                </>
              )}

              {!message.isThinking && message.runId && (message.shareUrl || message.usageMetadata) && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onFeedback(message.id, "positive", feedbackComment[message.id])}
                    aria-pressed={message.feedback === "positive"}
                    className={`h-8 px-2 text-xs rounded-md transition-all duration-150 ease-out active:scale-95 bg-transparent ${
                      message.feedback === "positive"
                        ? "text-white"
                        : "text-muted-foreground hover:text-black dark:text-white/70 dark:hover:text-black"
                    }`}
                  >
                    <ThumbsUp
                      className="w-3 h-3 mr-1 transition-transform duration-150"
                      aria-hidden="true"
                      fill={message.feedback === "positive" ? "currentColor" : "none"}
                    />
                    Good
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onFeedback(message.id, "negative", feedbackComment[message.id])}
                    aria-pressed={message.feedback === "negative"}
                    className={`h-8 px-2 text-xs rounded-md transition-all duration-150 ease-out active:scale-95 bg-transparent ${
                      message.feedback === "negative"
                        ? "text-white"
                        : "text-muted-foreground hover:text-black dark:text-white/70 dark:hover:text-black"
                    }`}
                  >
                    <ThumbsDown
                      className="w-3 h-3 mr-1 transition-transform duration-150"
                      aria-hidden="true"
                      fill={message.feedback === "negative" ? "currentColor" : "none"}
                    />
                    Bad
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onToggleComment(message.id)}
                    className="h-8 px-2 text-xs rounded-md transition-colors duration-150 ease-out text-muted-foreground hover:text-black dark:text-white/70 dark:hover:text-black"
                  >
                    <MessageSquare className="w-3 h-3 mr-1" />
                    Feedback
                  </Button>
                </>
              )}

            </div>

            {showCommentInput === message.id && (
              <div className="mt-2 w-full">
                <Textarea
                  value={feedbackComment[message.id] || ""}
                  onChange={(e) => {
                    setFeedbackComment((prev) => ({ ...prev, [message.id]: e.target.value }))
                  }}
                  placeholder="Add feedback about this response..."
                  className="min-h-[60px] text-xs"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      if (feedbackComment[message.id]?.trim() && message.feedback) {
                        onSubmitComment(message.id)
                      }
                    } else if (e.key === "Escape") {
                      onCancelComment(message.id)
                    }
                  }}
                />
                {!message.feedback && (
                  <p className="text-[10px] text-muted-foreground mt-1">
                    Select thumbs up or down before submitting
                  </p>
                )}
              </div>
            )}

            {showToolCalls && message.toolCalls && message.toolCalls.length > 0 && (
              <div className="mt-3 space-y-2">
                {message.toolCalls.map((tool) => (
                  <div
                    key={tool.id}
                    className="px-3 py-2 rounded-lg border border-border bg-muted/50 text-xs"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-primary">
                        Tool: {tool.name}
                      </span>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground">
                      <details>
                        <summary className="cursor-pointer hover:opacity-80">
                          View arguments
                        </summary>
                        <pre className="mt-1 whitespace-pre-wrap break-words text-[10px]">
                          {JSON.stringify(tool.args, null, 2)}
                        </pre>
                      </details>
                      {tool.output && (
                        <details className="mt-2">
                          <summary className="cursor-pointer hover:opacity-80">
                            View output
                          </summary>
                          <pre className="mt-1 whitespace-pre-wrap break-words text-[10px]">
                            {typeof tool.output === "string"
                              ? tool.output
                              : JSON.stringify(tool.output, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {message.subgraphOutputs && message.subgraphOutputs.length > 0 && (
              <div className="mt-3">
                <details className="group">
                  <summary className="cursor-pointer text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2">
                    <span>Subagent Outputs ({message.subgraphOutputs.length})</span>
                    <span className="text-[10px] opacity-50">Click to expand</span>
                  </summary>
                  <div className="mt-2 space-y-2">
                    {message.subgraphOutputs.map((subgraph, idx) => (
                      <details
                        key={`${subgraph.name}-${idx}`}
                        className="px-3 py-2 rounded-lg border border-primary/30 bg-primary/5"
                      >
                        <summary className="cursor-pointer flex items-center gap-2 text-xs hover:opacity-80">
                          <Settings className="w-3 h-3 text-primary" />
                          <span className="font-semibold text-primary">{subgraph.name}</span>
                          {subgraph.isStreaming && (
                            <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                              Running...
                            </span>
                          )}
                          {subgraph.isComplete && (
                            <span className="text-[10px] text-green-600 dark:text-green-400">
                              Complete
                            </span>
                          )}
                        </summary>
                        {subgraph.output ? (
                          <div className="mt-2 text-xs font-mono text-muted-foreground">
                            <pre className="whitespace-pre-wrap break-words text-[10px] max-h-60 overflow-y-auto">
                              {subgraph.output}
                            </pre>
                          </div>
                        ) : (
                          <div className="mt-2 text-[10px] text-muted-foreground italic">
                            Waiting for output...
                          </div>
                        )}
                      </details>
                    ))}
                  </div>
                </details>
              </div>
            )}

          </>
        )}
      </div>
    </div>
    </>
  )
}, (prevProps, nextProps) => {
  // Custom comparison: skip re-render only if props affecting THIS message are unchanged
  // Message content/object changed - always re-render (e.g., during streaming)
  if (prevProps.message !== nextProps.message) {
    return false
  }
  
  // copiedId changed - only re-render if it affects this message
  const copiedIdAffectsThis = 
    prevProps.copiedId !== nextProps.copiedId &&
    (prevProps.copiedId === prevProps.message.id || nextProps.copiedId === nextProps.message.id)
  
  // showCommentInput changed - only re-render if it affects this message  
  const commentInputAffectsThis =
    prevProps.showCommentInput !== nextProps.showCommentInput &&
    (prevProps.showCommentInput === prevProps.message.id || nextProps.showCommentInput === nextProps.message.id)
  
  // feedbackComment changed for THIS message
  const feedbackCommentChanged = 
    prevProps.feedbackComment[prevProps.message.id] !== nextProps.feedbackComment[nextProps.message.id]
  
  // Other props that affect rendering
  const otherPropsChanged =
    prevProps.showToolCalls !== nextProps.showToolCalls ||
    prevProps.isRegenerating !== nextProps.isRegenerating ||
    prevProps.isLastAssistant !== nextProps.isLastAssistant
  
  // Re-render if any relevant prop changed
  if (copiedIdAffectsThis || commentInputAffectsThis || feedbackCommentChanged || otherPropsChanged) {
    return false
  }
  
  // Function references - if they changed, we need to re-render (shouldn't happen with useCallback)
  const functionsChanged =
    prevProps.onCopy !== nextProps.onCopy ||
    prevProps.onRegenerate !== nextProps.onRegenerate ||
    prevProps.onEditAndRerun !== nextProps.onEditAndRerun ||
    prevProps.onFeedback !== nextProps.onFeedback ||
    prevProps.onSubmitComment !== nextProps.onSubmitComment ||
    prevProps.onCancelComment !== nextProps.onCancelComment ||
    prevProps.onToggleComment !== nextProps.onToggleComment ||
    prevProps.setFeedbackComment !== nextProps.setFeedbackComment
  
  if (functionsChanged) {
    return false
  }
  
  // All props that matter for this message are unchanged - skip re-render
  return true
})
