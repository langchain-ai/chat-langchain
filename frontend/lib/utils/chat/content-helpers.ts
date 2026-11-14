/**
 * Content Extraction and Formatting Utilities
 *
 * Functions for extracting and formatting message content from various formats.
 */

/**
 * Extract text content from various message content formats.
 * Handles:
 * - String content
 * - Array content with text objects
 * - Mixed content types
 */
export const extractTextFromContent = (content: any): string => {
  if (typeof content === "string") return content

  if (Array.isArray(content)) {
    return content
      .filter((c: any) => typeof c === "string" || c?.type === "text")
      .map((c: any) => (typeof c === "string" ? c : c.text || ""))
      .join("\n\n")
  }

  return ""
}

/** Base64 contains UTF-8 bytes; atob alone produces mojibake for non-ASCII files. */
export const decodeTextAttachment = (base64: string): string =>
  new TextDecoder().decode(Uint8Array.from(atob(base64), (character) => character.charCodeAt(0)))


const DOCS_BASE = "https://docs.langchain.com"
// Docs sections the agent cites; a site-relative link to one of these opens
// chat.langchain.com/<path> (404), so point it at the docs site instead.
const RELATIVE_DOCS_LINK = /\]\(\/((?:oss|langsmith|labs)\/[^)\s]*)\)/g

/** Rewrite site-relative docs links in markdown to docs.langchain.com, leaving code fences alone. */
export const absolutizeDocsLinks = (markdown: string): string =>
  markdown
    .split("```")
    .map((part, index) =>
      index % 2 === 0 ? part.replace(RELATIVE_DOCS_LINK, `](${DOCS_BASE}/$1)`) : part
    )
    .join("```")
