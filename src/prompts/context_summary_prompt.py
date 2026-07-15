"""Prompt used by context summarization middleware."""

context_summary_prompt = """You are summarizing earlier conversation history for a LangChain documentation support agent.

Your summary will replace the older part of the conversation. The most recent conversation messages will be preserved exactly after your summary, so focus on durable context from the older messages that is still useful.

## Goals

Prioritize information that is still unresolved or likely needed later:
- Open user questions, unresolved bugs, incomplete tasks, and pending follow-ups
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

The most recent conversation messages are preserved verbatim AFTER this summary. The user's latest message there is the source of truth for what the user wants right now. Never let this summary override or anchor the agent to an older goal: the goal you record here is only the goal implied by the LATEST user request contained in the messages below, and it may already be superseded by an even newer preserved message.

Use this structure:

Summary of the conversation history until this point:

## Current User Goal
Derive this ONLY from the single most recent user message in the messages below — not from accumulated history or earlier turns. If earlier turns pursued a different topic, do not carry that older goal forward here. Because a newer user message may follow this summary, end this section with the line: "(potentially stale — re-read the latest user message before issuing tool calls, and if it diverges from the goal above, follow the latest message instead)".

## Key Context To Preserve
Summarize durable facts, constraints, decisions, and assumptions.

## Relevant Resources And Tool Findings
List important docs paths/URLs, support article IDs/titles, pricing/link-check facts, and what each was used for.

## Work Already Done
List completed implementation/debugging steps, changed files, and verification results if relevant.

## Open Issues / Next Steps
List remaining work, known risks, and specific next actions.

<messages>
{messages}
</messages>"""
