export interface ChatUIState {
  input: string
  copiedId: string | null
  isLoading: boolean
  isLoadingThread: boolean
  isRegenerating: boolean
  isStopping: boolean
  hasAutoSent: boolean
  errorMessage: string | null
}

export type ChatUIAction =
  | { type: "SET_INPUT"; payload: string }
  | { type: "SET_COPIED_ID"; payload: string | null }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_LOADING_THREAD"; payload: boolean }
  | { type: "SET_STOPPING"; payload: boolean }
  | { type: "SET_AUTO_SENT"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "START_SEND" }
  | { type: "FINISH_SEND" }
  | { type: "START_REGENERATE" }
  | { type: "FINISH_REGENERATE" }

export function chatUIReducer(
  state: ChatUIState,
  action: ChatUIAction
): ChatUIState {
  switch (action.type) {
    case "SET_INPUT":
      return { ...state, input: action.payload }
    case "SET_COPIED_ID":
      return { ...state, copiedId: action.payload }
    case "SET_LOADING":
      return { ...state, isLoading: action.payload }
    case "SET_LOADING_THREAD":
      return { ...state, isLoadingThread: action.payload }
    case "SET_STOPPING":
      return { ...state, isStopping: action.payload }
    case "SET_AUTO_SENT":
      return { ...state, hasAutoSent: action.payload }
    case "SET_ERROR":
      return { ...state, errorMessage: action.payload }
    // Sending the current draft clears it in handleSend. Starting a queued run or
    // an initial URL prompt must preserve a different draft being composed.
    case "START_SEND":
      return { ...state, isLoading: true }
    case "FINISH_SEND":
      return { ...state, isLoading: false, isStopping: false }
    case "START_REGENERATE":
      return { ...state, isRegenerating: true }
    case "FINISH_REGENERATE":
      return { ...state, isRegenerating: false, isStopping: false }
    default:
      return state
  }
}

interface TextAttachment {
  mimeType: string
  textLength?: number
}

export function getAttachmentMimeType(file: {
  type: string
  name: string
}): string {
  if (file.type && file.type !== "application/octet-stream") return file.type
  const extension = file.name.split(".").pop()?.toLowerCase()
  const imageTypes: Record<string, string> = {
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    png: "image/png",
    gif: "image/gif",
    webp: "image/webp",
  }
  return (extension && imageTypes[extension]) || file.type || "text/plain"
}

export function getAttachedTextLength(
  files: readonly TextAttachment[]
): number {
  return files.reduce(
    (total, file) =>
      total + (file.mimeType.startsWith("image/") ? 0 : (file.textLength ?? 0)),
    0
  )
}

export function limitComposerInput(
  value: string,
  attachedTextLength: number,
  maxLength: number
) {
  const remaining = Math.max(0, maxLength - attachedTextLength)
  return {
    value: value.slice(0, remaining),
    overLimit: value.length > remaining,
  }
}

/** Recheck after asynchronous conversion using the current draft and attachments. */
export function appendWithinInputBudget<T extends TextAttachment>(
  files: readonly T[],
  nextFile: T,
  inputLength: number,
  maxLength: number
): T[] | null {
  if (
    inputLength +
      getAttachedTextLength(files) +
      getAttachedTextLength([nextFile]) >
    maxLength
  )
    return null
  return [...files, nextFile]
}

/** A synchronous claim survives effect replay before the reducer update renders. */
export function claimInitialPrompt(
  claimedThreads: Set<string>,
  threadId: string,
  initialMessage: string | null | undefined,
  ready: boolean
): string | null {
  const prompt = initialMessage?.trim()
  if (!ready || !prompt || claimedThreads.has(threadId)) return null
  claimedThreads.add(threadId)
  return prompt
}
