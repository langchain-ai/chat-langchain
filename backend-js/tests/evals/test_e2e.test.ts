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
import { Client, Run, Example } from 'langsmith'
import {
  evaluate as evaluateLangSmith,
  EvaluationResult,
  type EvaluateOptions,
} from 'langsmith/evaluation'
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
const JUDGE_MODEL_NAME = 'openai/gpt-4o-mini'

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
 * Matches Python signature: evaluate_retrieval_recall(run: Run, example: Example) -> dict
 */
function evaluateRetrievalRecall(
  run: Run,
  example?: Example,
): EvaluationResult {
  if (!example) {
    return { key: SCORE_RETRIEVAL_RECALL, score: 0.0 }
  }

  const expectedSources = example.metadata?.metadata?.source
  const retrievedSources = new Set<string>(
    run.outputs?.documents?.map?.((doc: Document) => doc.metadata.source) || [],
  )

  // Calculate recall - at least one expected source should be in retrieved docs
  const score = retrievedSources.has(expectedSources) ? 1.0 : 0.0

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
 * Matches Python signature: evaluate_qa(run: Run, example: Example) -> dict
 */
async function evaluateQA(
  run: Run,
  example?: Example,
): Promise<EvaluationResult> {
  if (!example) {
    return { key: SCORE_ANSWER_CORRECTNESS, score: 0.0 }
  }

  const messages = (run.outputs?.messages as any[]) || []
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
    question: example.inputs?.question,
    true_answer: example.outputs?.answer,
    answer: lastMessage.content,
  })) as GradeAnswer

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
 * Matches Python signature: evaluate_qa_context(run: Run, example: Example) -> dict
 */
async function evaluateQAContext(
  run: Run,
  example?: Example,
): Promise<EvaluationResult> {
  if (!example) {
    return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: 0.0 }
  }

  const messages = (run.outputs?.messages as any[]) || []
  if (messages.length === 0) {
    return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: 0.0 }
  }

  const documents = (run.outputs?.documents as Document[]) || []
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
    question: example.inputs?.question,
    context,
    answer: lastMessage.content,
  })) as GradeAnswer

  return { key: SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, score: result.score }
}

/**
 * Run the graph for evaluation
 * Wrapper to match LangSmith evaluate target signature
 */
async function runGraph(
  inputs: Record<string, any>,
): Promise<Record<string, any>> {
  const results = await graph.invoke({
    messages: [new HumanMessage(inputs.question)],
  })
  return results
}

/**
 * Main evaluation test
 * Uses LangSmith evaluate function (equivalent to Python's aevaluate)
 */
describe('E2E Evaluation Tests', () => {
  it('should pass regression test with minimum score thresholds', async () => {
    console.log('Starting evaluation...')

    const options: EvaluateOptions = {
      data: DATASET_NAME,
      evaluators: [evaluateQA, evaluateQAContext, evaluateRetrievalRecall],
      experimentPrefix: EXPERIMENT_PREFIX,
      metadata: { judge_model_name: JUDGE_MODEL_NAME },
      maxConcurrency: 1,
      client,
    }
    const { results: experimentResults } = await evaluateLangSmith(
      runGraph,
      options,
    )

    // Collect results and scores
    const results: Array<{
      input: any
      expectedOutput: any
      actualOutput: any
      scores: Record<string, number>
    }> = []

    // Process results as they become available
    for await (const result of experimentResults) {
      const scores: Record<string, number> = {}
      for (const evalResult of result.evaluationResults.results) {
        if (evalResult.score !== null && evalResult.score !== undefined) {
          scores[evalResult.key] = evalResult.score as number
        }
      }

      results.push({
        input: result.example.inputs,
        expectedOutput: result.example.outputs,
        actualOutput: result.run.outputs,
        scores,
      })
    }

    // Log all results in a table format
    const tableData = results.map((result, index) => ({
      Test: index + 1,
      Question: (result.input.question as string).substring(0, 50) + '...',
      [SCORE_RETRIEVAL_RECALL]:
        result.scores[SCORE_RETRIEVAL_RECALL]?.toFixed(2) ?? 'N/A',
      [SCORE_ANSWER_CORRECTNESS]:
        result.scores[SCORE_ANSWER_CORRECTNESS]?.toFixed(2) ?? 'N/A',
      [SCORE_ANSWER_VS_CONTEXT_CORRECTNESS]:
        result.scores[SCORE_ANSWER_VS_CONTEXT_CORRECTNESS]?.toFixed(2) ?? 'N/A',
    }))
    console.log('Records:')
    console.table(tableData)

    // Calculate average scores
    const avgAnswerCorrectness =
      results.reduce(
        (sum, r) => sum + (r.scores[SCORE_ANSWER_CORRECTNESS] ?? 0),
        0,
      ) / results.length
    const avgContextCorrectness =
      results.reduce(
        (sum, r) => sum + (r.scores[SCORE_ANSWER_VS_CONTEXT_CORRECTNESS] ?? 0),
        0,
      ) / results.length

    const avgRetrievalRecall =
      results.reduce(
        (sum, r) => sum + (r.scores[SCORE_RETRIEVAL_RECALL] ?? 0),
        0,
      ) / results.length

    // Print averages in a console.table for better visibility
    console.log('Averages:')
    console.table({
      [SCORE_ANSWER_CORRECTNESS]: avgAnswerCorrectness.toFixed(2),
      [SCORE_ANSWER_VS_CONTEXT_CORRECTNESS]: avgContextCorrectness.toFixed(2),
      [SCORE_RETRIEVAL_RECALL]: avgRetrievalRecall.toFixed(2),
    })

    // Assert minimum thresholds
    expect(avgAnswerCorrectness).toBeGreaterThanOrEqual(0.9)
    expect(avgContextCorrectness).toBeGreaterThanOrEqual(0.7)
    expect(avgRetrievalRecall).toBeGreaterThanOrEqual(0.9)
  }, 300000) // 5 minute timeout for full evaluation
})
