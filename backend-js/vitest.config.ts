import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 60000, // 60 seconds for LLM calls
    hookTimeout: 30000,
    include: ['tests/**/*.test.ts'],
  },
})
