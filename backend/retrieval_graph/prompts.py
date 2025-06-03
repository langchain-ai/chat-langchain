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
    client.pull_prompt("margot-na/router").messages[0].prompt.template
)
GENERATE_QUERIES_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/chat-langchain-generate-queries-prompt")
    .messages[0]
    .prompt.template
)
MORE_INFO_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/more_info").messages[0].prompt.template
)
RESEARCH_PLAN_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/researcher").messages[0].prompt.template
)
GENERAL_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/irrelevant_response").messages[0].prompt.template
)
RESPONSE_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/synthesizer").messages[0].prompt.template
)
