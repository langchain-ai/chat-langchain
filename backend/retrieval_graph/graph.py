"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.researcher_graph.graph import graph as researcher_graph
from backend.retrieval_graph.state import AgentState, InputState, Router
from backend.utils import format_docs, load_chat_model
from dotenv import load_dotenv
import os
import warnings
from langsmith import Client

load_dotenv()

warnings.filterwarnings("ignore", category=ResourceWarning)

async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    # allow skipping the router for testing
    if state.router and state.router["logic"]:
        return {"router": state.router}

    configuration = AgentConfiguration.from_runnable_config(config)
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model = load_chat_model(configuration.query_model).with_structured_output(
        Router, **structured_output_kwargs
    )
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    response = cast(Router, await model.ainvoke(messages))
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "langchain":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model = load_chat_model(configuration.query_model).with_structured_output(
        Plan, **structured_output_kwargs
    )
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    response = cast(
        Plan, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]})
    )
    return {
        "steps": response["steps"],
        "documents": "delete",
        "query": state.messages[-1].content,
    }


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"

def hallucination_check_finished(state: AgentState) -> Literal["respond", "end"]:
    """Determine if the hallucination check is complete or if more research is needed.
    """
    if (state.detected_hallucination == False or state.retried_answer >= 2):
        return "end"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    # TODO: add a re-ranker here
    top_k = 20
    context = format_docs(state.documents[:top_k])
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response], "answer": response.content}

async def hallucination_check(state: AgentState, *, config: RunnableConfig) -> Literal["respond", "end"]:
    """Check if the response is hallucinated."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    top_k = 20
    context = format_docs(state.documents[:top_k])
    final_answer_to_check_for_hallucination = state.answer
    
    system_prompt = """You are an expert judge evaluating whether an AI assistant's response is grounded in the provided context documents.

Your task is to determine if the final answer is relevant and supported by the context provided. 

Context Documents:
{context}

Final Answer to Evaluate:
{answer}

Evaluate whether the final answer is:
1. Factually supported by the context documents
2. Does not contain information that contradicts the context
3. Does not include fabricated details not present in the context
4. Stays within the scope of information available in the context

Provide your assessment with:
- is_relevant: boolean indicating if the answer is properly grounded in the context
- reasoning: detailed explanation of your assessment
"""
    prompt = system_prompt.format(context=context, answer=final_answer_to_check_for_hallucination)
    # Create proper messages array - this was the issue!
    messages = [{"role": "user", "content": prompt}]
    class HallucinationResult(TypedDict):
        """Is the LLM's final response is relevant to the context."""
        is_relevant: bool
        reasoning: str
    model = model.with_structured_output(HallucinationResult)
    result = await model.ainvoke(messages)
    if result["is_relevant"]:
        return {"detected_hallucination": False, "retried_answer": 0}
    else:
        print("hallucination detected")
        return {"detected_hallucination": True, "retried_answer": state.retried_answer + 1}

# Define the graph. uncomment hallucination_check to enable hallucination check / reflection step
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(create_research_plan)
builder.add_node(conduct_research)
builder.add_node(respond)
# builder.add_node(hallucination_check)
builder.add_edge(START, "create_research_plan")
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
# builder.add_edge("respond", "hallucination_check")
# builder.add_conditional_edges("hallucination_check", hallucination_check_finished, {"end": END, "respond": "respond"})
# builder.add_edge("hallucination_check", END)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"

if __name__ == "__main__":
    import asyncio
    # async def main():
    #     # Simple test of the compiled graph
    #     sample_question = "What is LangGraph and how does it work?"
        
    #     print(f"Invoking graph with question: {sample_question}")
    #     print("-" * 50)
        
    #     result = await graph.ainvoke({
    #         "messages": [HumanMessage(content=sample_question)]
    #     })
        
    #     print("Graph execution completed!")
    #     # print(f"Final answer: {result['answer']}")
    #     # print(f"Number of documents retrieved: {len(result.get('documents', []))}")
    #     # print(f"Research steps completed: {len(result.get('steps', []))}")
        
    # asyncio.run(main())
    
    # evaluate the graph
    async def main():
        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        dataset="Chat LangChain Golden Dataset"
        conciseness_prompt = """You are a judge evaluating answer conciseness with reasonable standards that balance brevity with clarity and completeness.

INSTRUCTION: Mark an answer as concise if it communicates effectively without excessive verbosity.

BALANCED CRITERIA:
- Answer should be reasonably efficient compared to the reference answer
- Some redundant words or phrases are acceptable if they improve understanding
- Common filler words are acceptable in moderation if they aid natural flow
- Repetitive explanations are acceptable if they clarify complex concepts
- Examples and elaborations are acceptable if they add meaningful value
- Answer should be clear and accessible to the reader
- Prioritize effective communication over extreme brevity
- Allow for reasonable explanations that enhance comprehension

mark as NOT concise if there are any of the following issues. hold a high standard for conciseness but be fair:
- Extensive repetition of the same point without added value
- Very long introductory sections that don't contribute meaningfully
- Excessive use of filler phrases throughout the response
- Significantly longer than necessary without clear benefit
- Overly complex explanations where simple ones would suffice
- Word count is substantially inflated compared to reference without justification

SCORING APPROACH:
- Focus on overall communication effectiveness
- Be understanding of different writing styles
- Consider that clarity often requires some additional words
- Allow for explanatory content that helps reader understanding
- Mark as concise if answer communicates well without being wasteful
- Give benefit of the doubt when verbosity serves a purpose

REFERENCE ANSWER:
{reference_answer}

ANSWER TO EVALUATE:
{answer}

EVALUATION PROCESS:
1. Compare overall communication effectiveness of both answers
2. Look for significant instances of unnecessary verbosity
3. Check if the answer is substantially longer without clear benefit
4. Ask: "Is this answer excessively verbose, or does it communicate well?"
5. Consider if additional length serves clarity, completeness, or accessibility
6. Focus on whether the answer effectively serves the reader

Be generous in your assessment while still identifying truly verbose responses.

Respond with:
- is_concise: boolean (lean toward true for reasonably efficient communication)
- reasoning: explain your evaluation, focusing on communication effectiveness"""

        relevance_prompt = """You are an expert judge evaluating how well an AI-generated answer is grounded in and relevant to the provided source documents.

Your task is to score the relevance of the answer to the given documents on a scale of 1-10.

SCORING CRITERIA:
10: Answer is completely grounded in the documents, all key points are directly supported
9: Answer is very well grounded, with minor unsupported details
8: Answer is well grounded, most information comes from documents
7: Answer is mostly grounded, some key points supported by documents
6: Answer is moderately grounded, about half the content is supported
5: Answer has mixed relevance, some connection to documents but significant gaps
4: Answer has limited relevance, few connections to the provided documents
3: Answer is barely relevant, minimal connection to documents
2: Answer is mostly irrelevant, very little connection to documents
1: Answer is completely irrelevant or contradicts the documents

EVALUATION FACTORS:
- How much of the answer's content is directly supported by the documents?
- Does the answer contain information not present in the documents?
- Are the main claims in the answer backed by the source material?
- Does the answer stay within the scope of information available in the documents?
- Are there any contradictions between the answer and the documents?

SOURCE DOCUMENTS:
{documents}

ANSWER TO EVALUATE:
{answer}

Instructions:
1. Carefully examine how well the answer aligns with the provided documents
2. Identify which parts of the answer are supported by the documents
3. Note any unsupported claims or contradictions
4. Assign a score from 1-10 based on the criteria above
5. Provide detailed reasoning for your score

Respond with:
- relevance_score: integer from 1-10 indicating relevance level
- reasoning: detailed explanation of your assessment and score"""

        accuracy_prompt = """You are an expert judge evaluating the factual accuracy of AI-generated answers compared to reference answers.

Your task is to score the accuracy of the given answer compared to the reference answer on a scale of 1-10.

SCORING CRITERIA:
10: Answer is completely accurate, all facts and details perfectly match the reference
9: Answer is highly accurate, with only very minor discrepancies that don't affect meaning
8: Answer is very accurate, most facts correct with minor inaccuracies
7: Answer is mostly accurate, some correct facts but a few notable errors
6: Answer is moderately accurate, about half the facts are correct
5: Answer has mixed accuracy, some correct information but significant errors
4: Answer has limited accuracy, few correct facts with many errors
3: Answer is barely accurate, minimal correct information
2: Answer is mostly inaccurate, very few correct facts
1: Answer is completely inaccurate or contradicts the reference answer

EVALUATION FACTORS:
- Are the core facts and claims in the answer factually correct?
- Do the technical details match the reference answer?
- Are there any factual contradictions between the answers?
- Does the answer maintain the same level of precision as the reference?
- Are numerical values, names, dates, and specific details accurate?
- Does the answer convey the same essential information as the reference?

REFERENCE ANSWER (Ground Truth):
{reference_answer}

ANSWER TO EVALUATE:
{answer}

Instructions:
1. Compare the factual content of both answers
2. Identify any factual discrepancies or errors
3. Consider the significance of any inaccuracies
4. Assign a score from 1-10 based on overall factual accuracy
5. Provide detailed reasoning explaining your assessment

Respond with:
- accuracy_score: integer from 1-10 indicating accuracy level
- reasoning: detailed explanation of your assessment and score"""

        async def target(inputs: dict):
            result = await graph.ainvoke({"messages": [HumanMessage(content=inputs["input_question"])]})
            return {"answer": result["answer"], "documents": result["documents"]}

        async def conciseness(inputs: dict, outputs: dict, reference_outputs: dict):
            answer, reference_answer = outputs["answer"], reference_outputs["output_answer"]
            
            class ConcisenessResult(TypedDict):
                """Evaluation of answer conciseness."""
                is_concise: bool
                reasoning: str
            
            # Use the same model configuration as the main graph
            configuration = AgentConfiguration()
            model = load_chat_model(configuration.query_model).with_structured_output(
                ConcisenessResult
            )
            prompt = conciseness_prompt.format(
                reference_answer=reference_answer,
                answer=answer
            )
            
            messages = [{"role": "user", "content": prompt}]
            result = await model.ainvoke(messages)
            print(result["is_concise"], "is_concise from conciseness evaluator", type(result["is_concise"]), "type of is_concise")
            if result["is_concise"]:
                score = 1
            else:   
                score = 0
            print(score, "score from conciseness evaluator")
            return {"key": "is_concise", "score": score}

        async def answer_relevance_docs(inputs: dict, outputs: dict, reference_outputs: dict):
            answer, documents = outputs["answer"], outputs["documents"]
            
            class RelevanceResult(TypedDict):
                """Evaluation of answer relevance to documents."""
                relevance_score: int
                reasoning: str
            
            # Use the same model configuration as the main graph
            configuration = AgentConfiguration()
            model = load_chat_model(configuration.query_model).with_structured_output(
                RelevanceResult
            )
            
            # Format documents for the prompt
            formatted_docs = format_docs(documents[:10])  # Use top 10 documents
            
            prompt = relevance_prompt.format(
                documents=formatted_docs,
                answer=answer
            )
            
            messages = [{"role": "user", "content": prompt}]
            result = await model.ainvoke(messages)
            
            return {"key": "answer_relevance_docs", "score": result["relevance_score"]}
        
        async def accuracy(inputs: dict, outputs: dict, reference_outputs: dict):
            answer, reference_answer = outputs["answer"], reference_outputs["output_answer"]
            
            class AccuracyResult(TypedDict):
                """Evaluation of answer accuracy."""
                accuracy_score: int
                reasoning: str
            
            # Use the same model configuration as the main graph
            configuration = AgentConfiguration()
            model = load_chat_model(configuration.query_model).with_structured_output(
                AccuracyResult
            )
            prompt = accuracy_prompt.format(
                reference_answer=reference_answer,
                answer=answer
            )
            messages = [{"role": "user", "content": prompt}]
            result = await model.ainvoke(messages)
            
            return {"key": "accuracy", "score": result["accuracy_score"]}

        experiment_results = await client.aevaluate(
            target,
            data="Chat LangChain Golden Dataset",
            evaluators=[conciseness, answer_relevance_docs, accuracy],
            experiment_prefix="gpt-4.1-no-reflection-step",
            max_concurrency=2,
            num_repetitions=3
        )

    asyncio.run(main())