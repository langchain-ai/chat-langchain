from langsmith import Client

"""Default prompts."""

client = Client()
# # fetch from langsmith
# INPUT_GUARDRAIL_SYSTEM_PROMPT = (
#     client.pull_prompt("input_guardrail")
#     .messages[0]
#     .prompt.template
# )

ROUTER_SYSTEM_PROMPT = (
    client.pull_prompt("router")
    .messages[0]
    .prompt.template
)
GENERATE_QUERIES_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-generate-queries-prompt")
    .messages[0]
    .prompt.template
)
MORE_INFO_SYSTEM_PROMPT = (
    client.pull_prompt("more_info")
    .messages[0]
    .prompt.template
)
RESEARCH_PLAN_SYSTEM_PROMPT = (
    client.pull_prompt("researcher")
    .messages[0]
    .prompt.template
)
GENERAL_SYSTEM_PROMPT = (
    client.pull_prompt("irrelevant_response")
    .messages[0]
    .prompt.template
)
RESPONSE_SYSTEM_PROMPT = (
    client.pull_prompt("synthesizer")
    .messages[0]
    .prompt.template
)
