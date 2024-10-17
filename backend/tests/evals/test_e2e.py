import asyncio
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResults, aevaluate
from langsmith.schemas import Example, Run

from backend.retrieval_graph.graph import graph
from backend.utils import format_docs

DATASET_NAME = "chat-langchain-qa"
EXPERIMENT_PREFIX = "chat-langchain-ci"

SCORE_RETRIEVAL_RECALL = "retrieval_recall"
SCORE_ANSWER_CORRECTNESS = "answer_correctness_score"
SCORE_ANSWER_VS_CONTEXT_CORRECTNESS = "answer_vs_context_correctness_score"

# claude sonnet / gpt-4o are a bit too expensive
JUDGE_MODEL_NAME = "gpt-4o-mini"

judge_llm = ChatOpenAI(model_name=JUDGE_MODEL_NAME)


# Evaluate retrieval


def evaluate_retrieval_recall(run: Run, example: Example) -> dict:
    documents: list[Document] = run.outputs.get("documents") or []
    sources = [doc.metadata["source"] for doc in documents]
    expected_sources = set(example.outputs.get("sources") or [])
    # NOTE: since we're currently assuming only ~1 correct document per question
    # this score is equivalent to recall @K where K is number of retrieved documents
    score = float(any(source in expected_sources for source in sources))
    return {"key": SCORE_RETRIEVAL_RECALL, "score": score}


# QA Evaluation Schema


class GradeAnswer(BaseModel):
    """Evaluate correctness of the answer and assign a continuous score."""

    reason: str = Field(
        description="1-2 short sentences with the reason why the score was assigned"
    )
    score: float = Field(
        description="Score that shows how correct the answer is. Use 1.0 if completely correct and 0.0 if completely incorrect",
        minimum=0.0,
        maximum=1.0,
    )


# Evaluate the answer based on the reference answers


QA_SYSTEM_PROMPT = """You are an expert programmer and problem-solver, tasked with grading answers to questions about Langchain.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        (
            "human",
            "QUESTION: \n\n {question} \n\n TRUE ANSWER: {true_answer} \n\n STUDENT ANSWER: {answer}",
        ),
    ]
)

qa_chain = QA_PROMPT | judge_llm.with_structured_output(GradeAnswer)


def evaluate_qa(run: Run, example: Example) -> dict:
    messages = run.outputs.get("messages") or []
    if not messages:
        return {"score": 0.0}

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return {"score": 0.0}

    score: GradeAnswer = qa_chain.invoke(
        {
            "question": example.inputs["question"],
            "true_answer": example.outputs["answer"],
            "answer": last_message.content,
        }
    )
    return {"key": SCORE_ANSWER_CORRECTNESS, "score": float(score.score)}


# Evaluate the answer based on the provided context

CONTEXT_QA_SYSTEM_PROMPT = """You are an expert programmer and problem-solver, tasked with grading answers to questions about Langchain.
You are given a question, the context for answering the question, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Grade the student answer BOTH based on its factual accuracy AND on whether it is supported by the context. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements."""

CONTEXT_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXT_QA_SYSTEM_PROMPT),
        (
            "human",
            "QUESTION: \n\n {question} \n\n CONTEXT: {context} \n\n STUDENT ANSWER: {answer}",
        ),
    ]
)

context_qa_chain = CONTEXT_QA_PROMPT | judge_llm.with_structured_output(GradeAnswer)


def evaluate_qa_context(run: Run, example: Example) -> dict:
    messages = run.outputs.get("messages") or []
    if not messages:
        return {"score": 0.0}

    documents = run.outputs.get("documents") or []
    if not documents:
        return {"score": 0.0}

    context = format_docs(documents)

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return {"score": 0.0}

    score: GradeAnswer = context_qa_chain.invoke(
        {
            "question": example.inputs["question"],
            "context": context,
            "answer": last_message.content,
        }
    )
    return {"key": SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, "score": float(score.score)}


# Run evaluation


async def run_graph(inputs: dict[str, Any]) -> dict[str, Any]:
    results = await graph.ainvoke(
        {
            "messages": [("human", inputs["question"])],
        }
    )
    return results


# Check results


def convert_single_example_results(evaluation_results: EvaluationResults):
    converted = {}
    for r in evaluation_results["results"]:
        converted[r.key] = r.score
    return converted


# NOTE: this is more of a regression test
def test_scores_regression():
    # test most commonly used model
    experiment_results = asyncio.run(
        aevaluate(
            run_graph,
            data=DATASET_NAME,
            evaluators=[evaluate_retrieval_recall, evaluate_qa, evaluate_qa_context],
            experiment_prefix=EXPERIMENT_PREFIX,
            metadata={"judge_model_name": JUDGE_MODEL_NAME},
            max_concurrency=4,
        )
    )
    experiment_result_df = pd.DataFrame(
        convert_single_example_results(result["evaluation_results"])
        for result in experiment_results._results
    )
    average_scores = experiment_result_df.mean()

    assert average_scores[SCORE_RETRIEVAL_RECALL] >= 0.65
    assert average_scores[SCORE_ANSWER_CORRECTNESS] >= 0.9
    assert average_scores[SCORE_ANSWER_VS_CONTEXT_CORRECTNESS] >= 0.9
