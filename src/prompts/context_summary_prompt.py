"""Prompt used by context summarization middleware."""

context_summary_prompt = """You are summarizing earlier conversation history for a LangChain documentation support agent.

Your summary will replace the older part of the conversation. The most recent conversation messages will be preserved exactly after your summary, so focus on durable context from the older messages that is still useful.

## Goals

Prioritize information that is still unresolved or likely needed later:
- Unresolved bugs, incomplete tasks, and pending follow-ups relevant to the latest human message; earlier unanswered requests were cancelled
- Decisions already made, constraints, assumptions, and user preferences
- Important code paths, repo/file names, APIs, error messages, commands, and configuration values
- Relevant docs/support resources already discovered
- Tool failures, contradictions, or gotchas that would be expensive or confusing to rediscover

Do not preserve every raw tool output. Many tool results can be re-fetched:
- Documentation search results are cached by normalized query/page size/language, but only include snippets/titles/links.
- Support article data is cached in-process and article content can be re-read by article ID.
- Pricing data is cached briefly but should be re-fetched live before final pricing answers.
- Link checks are cached by URL.

Prefer compact references over full copied content:
- Keep docs page paths/URLs, support article IDs, titles, and the specific fact they supported.
- Keep exact error messages or command outputs only when they are central to unresolved work.
- Summarize repeated or resolved tool calls briefly.

If earlier messages included images or files, summarize only the useful interpretation or filename/context. Do not include base64 data or raw large file contents.

## Output Requirements

Try to keep the summary under 30,000 tokens.
Be concise but complete enough for the agent to continue without re-reading the removed history.
Respond only with the summary. Do not add meta-commentary.
The first line must begin exactly with: "Summary of the conversation history until this point:"

Use this structure:

Summary of the conversation history until this point:

## Current User Goal
Take the current user goal only from the latest human message, not earlier unanswered requests. If the latest human message is outside the history being summarized, leave the current goal to the preserved messages.

## Key Context To Preserve
Summarize durable facts, constraints, decisions, and assumptions.

## Relevant Resources And Tool Findings
List important docs paths/URLs, support article IDs/titles, pricing/link-check facts, and what each was used for.

## Work Already Done
List completed implementation/debugging steps, changed files, and verification results if relevant.

## Open Issues / Next Steps
List remaining work, known risks, and specific next actions relevant to the latest human message, not cancelled requests.

<messages>
{messages}
</messages>"""
