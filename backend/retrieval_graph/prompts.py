from langchain import hub

"""Default prompts."""

# fetch from langsmith
router_system_prompt = hub.pull("chat-langchain-router-prompt").messages[0].prompt.template
generate_queries_system_prompt = hub.pull("chat-langchain-generate-queries-prompt").messages[0].prompt.template
more_info_system_prompt = hub.pull("chat-langchain-more-info-prompt").messages[0].prompt.template
research_plan_system_prompt = hub.pull("chat-langchain-research-plan-prompt").messages[0].prompt.template
general_system_prompt = hub.pull("chat-langchain-general-prompt").messages[0].prompt.template
response_system_prompt = hub.pull("chat-langchain-response-prompt").messages[0].prompt.template

ROUTER_SYSTEM_PROMPT = router_system_prompt

GENERAL_SYSTEM_PROMPT = general_system_prompt

MORE_INFO_SYSTEM_PROMPT = more_info_system_prompt

RESEARCH_PLAN_SYSTEM_PROMPT = research_plan_system_prompt

RESPONSE_SYSTEM_PROMPT = response_system_prompt

GENERATE_QUERIES_SYSTEM_PROMPT = generate_queries_system_prompt
