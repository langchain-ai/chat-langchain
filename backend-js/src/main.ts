/**
 * Main entrypoint for testing the retrieval graph.
 *
 * This script provides a simple way to test the retrieval graph locally,
 * similar to the Python main.py file.
 */

// Load environment variables from .env file
import 'dotenv/config'

import { HumanMessage } from '@langchain/core/messages'
import { graph } from './retrieval_graph/graph.js'

/**
 * Test the graph with a sample query
 */
async function testGraph(): Promise<void> {
  try {
    console.log('🚀 Testing Retrieval Graph...\n')

    const result = await graph.invoke({
      messages: [new HumanMessage('How to connect LangChain to MCP?')],
    })

    console.log('✅ Graph execution completed!\n')
    console.log(`📝 Answer: ${result.answer || 'N/A'}\n`)
    console.log(`📚 Documents retrieved: ${result.documents?.length || 0}\n`)

    // Print a sample of documents if available
    if (result.documents && result.documents.length > 0) {
      console.log('📄 Sample documents:')
      result.documents.slice(0, 3).forEach((doc, idx) => {
        console.log(`\n  Document ${idx + 1}:`)
        console.log(`    Source: ${doc.metadata?.source || 'N/A'}`)
        console.log(`    Preview: ${doc.pageContent.substring(0, 100)}...`)
      })
    }
  } catch (error) {
    console.error('❌ Error executing graph:', error)
    if (error instanceof Error) {
      console.error('Error message:', error.message)
      console.error('Error stack:', error.stack)
    }
    process.exit(1)
  }
}

// Run the test
testGraph()
  .then(() => {
    console.log('\n✨ Test completed successfully!')
    process.exit(0)
  })
  .catch((error) => {
    console.error('\n💥 Test failed:', error)
    process.exit(1)
  })
