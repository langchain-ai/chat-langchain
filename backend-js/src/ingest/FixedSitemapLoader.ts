/**
 * Fixed version of SitemapLoader that correctly implements filterUrls
 * and provides better content extraction and metadata.
 *
 * This fixes two bugs in @langchain/community's SitemapLoader:
 * 1. The _checkUrlPatterns method has inverted logic
 * 2. The _loadSitemapUrls method extracts too much content and wrong metadata
 */

import { SitemapLoader } from '@langchain/community/document_loaders/web/sitemap'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { Document } from '@langchain/core/documents'
import { simpleExtractor, extractMetadata } from './parser.js'
import type { CheerioAPI } from 'cheerio'

interface SiteMapElement {
  loc: string
  changefreq?: string
  lastmod?: string
  priority?: string
}

interface FixedSitemapLoaderOptions {
  filterUrls?: (string | RegExp)[]
  chunkSize?: number
  extractor?: (html: string | CheerioAPI) => string
}

export class FixedSitemapLoader extends SitemapLoader {
  private customExtractor?: (html: string | CheerioAPI) => string

  constructor(webPath: string, options?: FixedSitemapLoaderOptions) {
    const { extractor, ...loaderOptions } = options || {}
    super(webPath, loaderOptions)
    this.customExtractor = extractor
  }
  /**
   * Check if a URL should be skipped based on allowUrlPatterns.
   *
   * Original buggy logic:
   * - Returns true when URL matches patterns (causing it to be skipped)
   * - Returns false when URL doesn't match (causing it to be included)
   *
   * Fixed logic:
   * - Returns true when URL doesn't match any pattern (should be skipped)
   * - Returns false when URL matches at least one pattern (should be included)
   *
   * @param url - The URL to check
   * @returns true if URL should be skipped, false if it should be included
   */
  _checkUrlPatterns(url: string): boolean {
    // If no filter patterns are set, don't skip any URLs
    if (!this.allowUrlPatterns) return false

    // Skip URL if it doesn't match ANY of the patterns
    // (i.e., return true only if NONE of the patterns match)
    return this.allowUrlPatterns.every(
      (pattern) => !new RegExp(pattern).test(url),
    )
  }

  /**
   * Load and extract content from sitemap URLs.
   *
   * This override fixes the original implementation which:
   * - Uses default selector that captures too much content
   * - Extracts metadata from wrong meta tags (og:title instead of title)
   *
   * New implementation:
   * - Uses custom extractor (if provided) or simpleExtractor for clean content extraction
   * - Extracts metadata properly (title from <title> tag, description, language)
   *
   * @param elements - Array of sitemap elements to load
   * @returns Array of Document objects with properly extracted content and metadata
   */
  async _loadSitemapUrls(elements: SiteMapElement[]): Promise<Document[]> {
    // Scrape all URLs from the sitemap
    const all = await CheerioWebBaseLoader.scrapeAll(
      elements.map((ele) => ele.loc),
      this.caller,
      this.timeout,
      this.textDecoder,
    )

    // Use custom extractor if provided, otherwise use simpleExtractor
    const extractor = this.customExtractor || simpleExtractor

    // Process each scraped page
    const documents = all.map(($, i) => {
      if (!elements[i]) {
        throw new Error('Scraped docs and elements not in sync')
      }

      // Use extractor to get clean text content (matches Python behavior)
      const text = extractor($)

      // Extract metadata from the HTML (matches Python metadata_extractor)
      const customMetadata = extractMetadata($)

      // Get sitemap-specific metadata
      const { loc: source, ...sitemapMetadata } = elements[i]

      return new Document({
        pageContent: text,
        metadata: {
          ...sitemapMetadata, // changefreq, lastmod, priority from sitemap
          source: source.trim(),
          ...customMetadata, // title, description, language from HTML
        },
      })
    })

    return documents
  }
}
