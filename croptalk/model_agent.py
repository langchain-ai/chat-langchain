from langchain.globals import set_debug
import os
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function

from croptalk.prompts_agent import agent_system_message
from croptalk.tools import tools

from dotenv import load_dotenv
load_dotenv()

set_debug(True)


def initialize_llm(model):

    return ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0,
    )


def initialize_llm_with_tools(model, tools):

    llm = initialize_llm(model)
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools])

    return llm_with_tools


def initialize_agent(model, tools, memory_key="chat_history", eval_mode=False):
    llm_with_tools = initialize_llm_with_tools(model, tools)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=agent_system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )

    agent = (
        {
            "input": itemgetter("question"),
            "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm_with_tools
    )

    if not eval_mode:
        agent = agent | OpenAIFunctionsAgentOutputParser()

    # TODO: use OpenAIFunctionsAgent instead of binding
    # openai_agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    return agent


def initialize_agent_executor(model, tools, memory_key="chat_history", memory_model="gpt-3.5-turbo-1106"):

    llm = initialize_llm(memory_model)

    memory = AgentTokenBufferMemory(
        memory_key=memory_key, llm=llm, max_token_limit=6000)
    agent = initialize_agent(model, tools, memory_key=memory_key)

    agent_executor = (
        AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True,
        ) | itemgetter("output")
    ).with_config(run_name="AgentExecutor")

    return agent_executor


model_name = os.getenv("MODEL_NAME")
model = initialize_agent_executor(model=model_name, tools=tools)
