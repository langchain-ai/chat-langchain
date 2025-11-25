/**
 * End-to-end evaluation tests for the retrieval graph.
 *
 * This module tests the complete system against a dataset of questions
 * and expected answers, measuring retrieval recall and answer correctness.
 */

import 'dotenv/config'

import { describe, it, expect } from 'vitest'
import { HumanMessage, AIMessage } from '@langchain/core/messages'
import { Document } from '@langchain/core/documents'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { Client } from 'langsmith'
import { z } from 'zod'
import { graph } from '../../src/retrieval_graph/graph.js'
import { formatDocs, loadChatModel } from '../../src/utils.js'

// Dataset and experiment configuration
const DATASET_NAME = 'small-chatlangchain-dataset'
const EXPERIMENT_PREFIX = 'chat-langchain-ci'

// Score keys
const SCORE_RETRIEVAL_RECALL = 'retrieval_recall'
const SCORE_ANSWER_CORRECTNESS = 'answer_correctness_score'
const SCORE_ANSWER_VS_CONTEXT_CORRECTNESS =
  'answer_vs_context_correctness_score'

// Judge model
const JUDGE_MODEL_NAME = 'groq/gpt-oss-20b'

const judgeModel = loadChatModel(JUDGE_MODEL_NAME)

// Initialize LangSmith client
const client = new Client()

/**
 * Schema for grading answers
 */
const GradeAnswerSchema = z.object({
  reason: z
    .string()
    .describe('1-2 short sentences with the reason why the score was assigned'),
  score: z
    .number()
    .min(0.0)
    .max(1.0)
    .describe(
      'Score that shows how correct the answer is. Use 1.0 if completely correct and 0.0 if completely incorrect',
    ),
})

type GradeAnswer = z.infer<typeof GradeAnswerSchema>

/**
 * Evaluate retrieval recall
 */
function evaluateRetrievalRecall(
  runOutput: any,
  trueSources: Set<string>,
): { key: string; score: number } {
  const documents: Document[] = runOutput.documents || []
  const sources = documents.map((doc) => doc.metadata.source)

  // Calculate recall - at least one expected source should be in retrieved docs
  const score = sources.some((source) => trueSources.has(source)) ? 1.0 : 0.0

  return { key: SCORE_RETRIEVAL_RECALL, score }
}

/**
 * QA evaluation system prompt
 */
const QA_SYSTEM_PROMPT = `You are an expert programmer and problem-solver, tasked with grading answers to questions about Langchain.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements.`

const QA_PROMPT = ChatPromptTemplate.fromMessages([
  ['system', QA_SYSTEM_PROMPT],
  [
    'human',
    'QUESTION: \n\n {question} \n\n TRUE ANSWER: {true_answer} \n\n STUDENT ANSWER: {answer}',
  ],
])

/**
 * Evaluate answer correctness based on reference answer
 */
async function evaluateQA(
  runOutput: any,
  runInput: any,
  expectedOutput: any,
): Promise<{ key: string; score: number }> {
  const messages = runOutput.messages || []
  if (messages.length === 0) {
    return { key: SCORE_ANSWER_CORRECTNESS, score: 0.0 }
  }

  const lastMessage = messages[messages.length - 1]
  if (!(lastMessage instanceof AIMessage)) {
    return { key: SCORE_ANSWER_CORRECTNESS, score: 0.0 }
  }

  const qaChain = QA_PROMPT.pipe(
    judgeModel.withStructuredOutput(GradeAnswerSchema),
  )

  const result = (await qaChain.invoke({
    question: runInput.question,
    true_answer: expectedOutput.answer,
    answer: lastMessage.content,
  })) as GradeAnswer
  console.log('🚀 ~ result:', result)

  return { key: SCORE_ANSWER_CORRECTNESS, score: result.score }
}

/**
 * Context QA evaluation system prompt
 */
const CONTEXT_QA_SYSTEM_PROMPT = `You are an expert programmer and problem-solver, tasked with grading answers to questions about Langchain.
You are given a question, the context for answering the question, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Grade the student answer BOTH based on its factual accuracy AND on whether it is supported by the context. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements.`

const CONTEXT_QA_PROMPT = ChatPromptTemplate.fromMessages([
  ['system', CONTEXT_QA_SYSTEM_PROMPT],
  [
    'human',
    'QUESTION: \n\n {question} \n\n CONTEXT: {context} \n\n STUDENT ANSWER: {answer}',
  ],
])

/**
 * Evaluate answer correctness based on retrieved context
 */
async function evaluateQAContext(
  runOutput: any,
  runInput: any,
): Promise<{ key: string; score: number }> {
  const messages = runOutput.messages || []
  if (messages.length === 0) {
    return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: 0.0 }
  }

  const documents = runOutput.documents || []
  if (documents.length === 0) {
    return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: 0.0 }
  }

  const context = formatDocs(documents)

  const lastMessage = messages[messages.length - 1]
  if (!(lastMessage instanceof AIMessage)) {
    return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: 0.0 }
  }

  const contextQaChain = CONTEXT_QA_PROMPT.pipe(
    judgeModel.withStructuredOutput(GradeAnswerSchema),
  )

  const result = (await contextQaChain.invoke({
    question: runInput.question,
    context,
    answer: lastMessage.content,
  })) as GradeAnswer

  return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: result.score }
}

/**
 * Run the graph for evaluation
 */
async function runGraph(inputs: { question: string }): Promise<any> {
  const results = await graph.invoke({
    messages: [new HumanMessage(inputs.question)],
  })
  return results
}

/**
 * Main evaluation test
 */
describe('E2E Evaluation Tests', () => {
  it('should pass regression test with minimum score thresholds', async () => {
    console.log('Starting evaluation...')

    // Load dataset from LangSmith
    const examples = []
    for await (const example of client.listExamples({
      datasetName: DATASET_NAME,
    })) {
      examples.push(example)
    }

    console.log(`Loaded ${examples.length} examples from dataset`)

    const results: Array<{
      input: any
      expectedOutput: any
      actualOutput: any
      scores: Record<string, number>
    }> = []

    const trueSources = new Set<string>(
      examples.map((example) => example.metadata?.metadata?.source),
    )

    // Run evaluation for each example
    for (const example of examples) {
      const actualOutput = await runGraph(
        example.inputs as { question: string },
      )

      const retrievalScore = await evaluateRetrievalRecall(
        actualOutput,
        trueSources,
      )

      // Evaluate
      const qaScore = await evaluateQA(
        actualOutput,
        example.inputs,
        example.outputs,
      )
      const contextScore = await evaluateQAContext(actualOutput, example.inputs)

      const scores = {
        [qaScore.key]: qaScore.score,
        [contextScore.key]: contextScore.score,
        [retrievalScore.key]: retrievalScore.score,
      }

      results.push({
        input: example.inputs,
        expectedOutput: example.outputs,
        actualOutput,
        scores,
      })

      console.log(`Scores:`, scores)
    }

    // Calculate average scores
    const avgAnswerCorrectness =
      results.reduce((sum, r) => sum + r.scores[SCORE_ANSWER_CORRECTNESS], 0) /
      results.length
    const avgContextCorrectness =
      results.reduce(
        (sum, r) => sum + r.scores[SCORE_ANSWER_VS_CONTEXT_CORRECTNESS],
        0,
      ) / results.length

    const avgRetrievalRecall =
      results.reduce((sum, r) => sum + r.scores[SCORE_RETRIEVAL_RECALL], 0) /
      results.length

    console.log(
      `\nAverage Answer Correctness: ${avgAnswerCorrectness.toFixed(2)}`,
    )
    console.log(
      `Average Context Correctness: ${avgContextCorrectness.toFixed(2)}`,
    )
    console.log(`Average Retrieval Recall: ${avgRetrievalRecall.toFixed(2)}`)

    // Assert minimum thresholds
    expect(avgAnswerCorrectness).toBeGreaterThanOrEqual(0.9)
    expect(avgContextCorrectness).toBeGreaterThanOrEqual(0.7)
    expect(avgRetrievalRecall).toBeGreaterThanOrEqual(0.9)
  }, 300000) // 5 minute timeout for full evaluation
})
