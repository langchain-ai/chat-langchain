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
  example: Example,
): { key: string; score: number } {
  const documents: Document[] = (run.outputs?.documents as Document[]) || []
  const sources = documents.map((doc) => doc.metadata.source)
  const expectedSources = new Set((example.outputs?.sources as string[]) || [])

  // Calculate recall - at least one expected source should be in retrieved docs
  const score = sources.some((source) => expectedSources.has(source))
    ? 1.0
    : 0.0

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
  example: Example,
): Promise<{ key: string; score: number }> {
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
  example: Example,
): Promise<{ key: string; score: number }> {
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
 */
async function runGraph(inputs: { question: string }): Promise<any> {
  const results = await graph.invoke({
    messages: [new HumanMessage(inputs.question)],
  })
  return results
}

/**
 * Main evaluation test
 * Uses LangSmith evaluation pattern similar to Python's aevaluate
 */
describe('E2E Evaluation Tests', () => {
  it('should pass regression test with minimum score thresholds', async () => {
    console.log('Starting evaluation...')

    // Load dataset from LangSmith
    const examples: Example[] = []
    for await (const example of client.listExamples({
      datasetName: DATASET_NAME,
    })) {
      examples.push(example)
    }

    console.log(`Loaded ${examples.length} examples from dataset`)

    // Create experiment name
    const experimentName = `${EXPERIMENT_PREFIX}-${new Date()
      .toISOString()
      .replace(/[:.]/g, '-')
      .slice(0, -5)}`

    const results: Array<{
      input: any
      expectedOutput: any
      actualOutput: any
      scores: Record<string, number>
    }> = []

    // Run evaluation for each example
    for (const example of examples) {
      // Run the graph
      const actualOutput = await runGraph(
        example.inputs as { question: string },
      )

      // Create a Run object for LangSmith tracking (similar to Python's aevaluate)
      const runId = crypto.randomUUID()
      const run: Run = {
        id: runId,
        name: 'run_graph',
        inputs: example.inputs as Record<string, any>,
        outputs: {
          messages: actualOutput.messages,
          documents: actualOutput.documents,
        } as Record<string, any>,
        run_type: 'chain',
        start_time: Date.now(),
        end_time: Date.now(),
      }

      // Evaluate in parallel using evaluators that match Python signature
      const [qaScore, contextScore, retrievalScore] = await Promise.all([
        evaluateQA(run, example),
        evaluateQAContext(run, example),
        Promise.resolve(evaluateRetrievalRecall(run, example)),
      ])

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

      // Log evaluation results to LangSmith (similar to Python's aevaluate)
      try {
        await client.createRun({
          id: runId,
          name: 'run_graph',
          inputs: example.inputs as Record<string, any>,
          outputs: {
            messages: actualOutput.messages,
            documents: actualOutput.documents,
          } as Record<string, any>,
          run_type: 'chain',
          start_time: Date.now(),
          end_time: Date.now(),
          project_name: experimentName,
          extra: {
            metadata: { judge_model_name: JUDGE_MODEL_NAME },
          },
        })

        // Log evaluation results as feedback
        for (const evalResult of [qaScore, contextScore, retrievalScore]) {
          await client.createFeedback(runId, evalResult.key, {
            score: evalResult.score,
            value: evalResult.score,
          })
        }
      } catch (error) {
        // Continue even if LangSmith logging fails
        console.warn('Failed to log to LangSmith:', error)
      }
    }

    // Log all results in a table format
    const tableData = results.map((result, index) => ({
      Test: index + 1,
      Question: (result.input.question as string).substring(0, 50) + '...',
      [SCORE_RETRIEVAL_RECALL]:
        result.scores[SCORE_RETRIEVAL_RECALL].toFixed(2),
      [SCORE_ANSWER_CORRECTNESS]:
        result.scores[SCORE_ANSWER_CORRECTNESS].toFixed(2),
      [SCORE_ANSWER_VS_CONTEXT_CORRECTNESS]:
        result.scores[SCORE_ANSWER_VS_CONTEXT_CORRECTNESS].toFixed(2),
    }))
    console.table(tableData)

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
    console.log(`\nLangSmith Experiment: ${experimentName}`)

    // Assert minimum thresholds
    expect(avgAnswerCorrectness).toBeGreaterThanOrEqual(0.9)
    expect(avgContextCorrectness).toBeGreaterThanOrEqual(0.7)
    expect(avgRetrievalRecall).toBeGreaterThanOrEqual(0.9)
  }, 300000) // 5 minute timeout for full evaluation
})
