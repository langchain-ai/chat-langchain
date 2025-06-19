"use client";

// This file centralizes all environment configuration with debugging
// We use 'use client' to ensure it works in client components

// Check if we're in browser environment
const isBrowser = typeof window !== 'undefined';

// RUNTIME PATCHING - Override any incorrect URLs
if (isBrowser) {
  // This will run in the browser and patch any hard-coded URL references
  const originalFetch = window.fetch;
  window.fetch = async function (input: RequestInfo | URL, init?: RequestInit) {
    let url: string;
    if (typeof input === "string") {
      url = input;
    } else if (input instanceof Request) {
      url = input.url;
    } else if (input instanceof URL) {
      url = input.toString();
    } else {
      url = String(input);
    }

    if (url.includes("localhost:2024")) {
      // Replace with the correct API URL
      const correctApiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL;
      
      if (correctApiUrl) {
        url = url.replace(/http:\/\/localhost:2024/g, correctApiUrl);
      } else {
        console.error('[FETCH] No API URL available for localhost replacement');
      }
    }

    const response = await originalFetch(url, init);
    if (!response.ok) {
      const snippet = await response.clone().text();
      console.error(
        `[FETCH] ${init?.method ?? "GET"} ${url} -> ${response.status}`,
        snippet.slice(0, 200)
      );
    }
    return response;
  };
}

export const ENV = {
  // Public API URL (accessible in client components)
  API_URL: (() => {
    // Prioritize NEXT_PUBLIC_API_URL since it's available in browser
    const url = process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL;
    if (!url) {
      console.error('[CONFIG] No API URL found in environment variables');
      throw new Error('API URL is required');
    }
    return url;
  })(),

  // API Base URL (accessible in client components via NEXT_PUBLIC prefix)
  API_BASE_URL: (() => {
    const url = process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL;
    if (!url) {
      console.error('[CONFIG] No API Base URL found in environment variables');
      throw new Error('API Base URL is required');
    }
    return url;
  })(),

  // LangChain API Key
  LANGCHAIN_API_KEY: (() => {
    const key = process.env.LANGCHAIN_API_KEY || '';
    return key;
  })(),

    // Debug function to print environment variables
  debug: () => {
    console.log('[CONFIG] Environment Variables:');
    console.log('  NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);
    console.log('  API_BASE_URL:', process.env.API_BASE_URL);
    console.log('  LANGCHAIN_API_KEY available:', !!process.env.LANGCHAIN_API_KEY);
    console.log('  NODE_ENV:', process.env.NODE_ENV);
    console.log('  CURSOR new attempt:');

    if (isBrowser) {
      console.log('  Running in browser at:', window.location.href);
    }
  }
};

// Print debug info on load
ENV.debug(); 

