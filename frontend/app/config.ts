"use client";

// Environment configuration for chat-langchain frontend

export const ENV = {
  API_URL: process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL || '',
  API_BASE_URL: process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL || '',
  LANGCHAIN_API_KEY: process.env.LANGCHAIN_API_KEY || '',
}; 

