/**
 * HTML parsing utilities for document ingestion.
 *
 * This module provides functions to extract text content from HTML documents,
 * with specific handling for LangChain documentation structure.
 */

import * as cheerio from 'cheerio'

/**
 * Extract text content from LangChain documentation HTML.
 *
 * This function extracts the main content from LangChain documentation pages,
 * focusing on the article content and removing unnecessary elements.
 *
 * @param html - HTML string or cheerio instance
 * @returns Extracted text content
 */
export function langchainDocsExtractor(
  html: string | cheerio.CheerioAPI,
): string {
  const $ = typeof html === 'string' ? cheerio.load(html) : html

  // Remove unwanted elements before extraction
  // This matches Python's behavior which uses SoupStrainer to filter during parsing
  // Remove navigation, menus, sidebars, and other non-content elements
  $(
    'script, style, nav, header, footer, iframe, noscript, form, button, ' +
      'aside, [role="navigation"], [role="banner"], [role="complementary"], ' +
      '[class*="sidebar"], [class*="menu"], [class*="nav"], ' +
      '[class*="toc"], [class*="breadcrumb"]',
  ).remove()

  // Try to find the main article content
  let content = ''

  // Look for common content selectors
  const articleSelector = $('article')
  if (articleSelector.length > 0) {
    // Remove any remaining navigation elements inside article
    articleSelector
      .find('nav, [role="navigation"], [class*="sidebar"], [class*="menu"]')
      .remove()
    content = articleSelector.text()
  } else {
    // Fallback to body if no article found
    content = $('body').text()
  }

  // Clean up the text
  return cleanText(content)
}

/**
 * Simple HTML text extractor.
 *
 * This function extracts all text content from HTML, removing all tags
 * and cleaning up whitespace.
 *
 * @param html - HTML string or cheerio instance
 * @returns Extracted text content
 */
export function simpleExtractor(html: string | cheerio.CheerioAPI): string {
  const $ = typeof html === 'string' ? cheerio.load(html) : html

  // Remove unwanted elements before extraction
  // This matches Python's behavior which uses SoupStrainer to filter during parsing
  // Remove navigation, menus, sidebars, and other non-content elements
  $(
    'script, style, nav, header, footer, iframe, noscript, form, button, ' +
      'aside, [role="navigation"], [role="banner"], [role="complementary"], ' +
      '[class*="sidebar"], [class*="menu"], [class*="nav"], ' +
      '[class*="toc"], [class*="breadcrumb"]',
  ).remove()

  // Extract only from article tag or main content area
  // This matches Python's: SoupStrainer(name=("article", "title", "html", "lang", "content"))
  let content = ''
  const article = $('article')
  if (article.length > 0) {
    // Remove any remaining navigation elements inside article
    article
      .find('nav, [role="navigation"], [class*="sidebar"], [class*="menu"]')
      .remove()
    content = article.text()
  } else {
    // Fallback: try main content area
    const main = $('main')
    if (main.length > 0) {
      main
        .find('nav, [role="navigation"], [class*="sidebar"], [class*="menu"]')
        .remove()
      content = main.text()
    } else {
      content = $('body').text()
    }
  }

  return cleanText(content)
}

/**
 * Clean extracted text by normalizing whitespace.
 *
 * @param text - Raw text to clean
 * @returns Cleaned text
 */
function cleanText(text: string): string {
  return (
    text
      // Replace multiple newlines with double newline
      .replace(/\n\n+/g, '\n\n')
      // Replace multiple spaces with single space
      .replace(/[ \t]+/g, ' ')
      // Trim whitespace from each line
      .split('\n')
      .map((line) => line.trim())
      .join('\n')
      // Remove empty lines at start and end
      .trim()
  )
}

/**
 * Extract metadata from HTML page.
 *
 * @param html - HTML string or cheerio instance
 * @param titleSuffix - Optional suffix to append to title
 * @returns Metadata object with title, description, and language
 */
export function extractMetadata(
  html: string | cheerio.CheerioAPI,
  titleSuffix?: string,
): Record<string, string> {
  const $ = typeof html === 'string' ? cheerio.load(html) : html

  let title = $('title').text() || ''
  if (titleSuffix) {
    title += titleSuffix
  }

  const description = $('meta[name="description"]').attr('content') || ''
  const language = $('html').attr('lang') || ''

  return {
    title,
    description,
    language,
  }
}
