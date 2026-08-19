# Scope marker: unsubstituted `{memory_path}` in the Fleet deep-agent system prompt

LangSmith issue: `7de1dd96-1ecc-4c87-9f91-f409647671bc` (project `fleet`)

**This defect is not in `chat-langchain`.** It lives in the Fleet backend service that
renders the deep-agent platform system prompt. This note exists only to record the
diagnosis for whoever picks the issue up; no `chat-langchain` behaviour changes.

## What the traces show

Every sampled LLM span in project `fleet` receives a system prompt whose
`<additional_important_instructions_to_remember>` block contains the literal string
`{memory_path}` — 19 occurrences per prompt, in 4 of 4 sampled tenant agents across
three templates. Evidence spans:

- trace `01a0157c-a232-7560-aa13-b774c932ccab`, run `01a0157c-ab88-7ce0-9e83-2f772e747243`
- trace `01a01692-7f1f-7a53-baa8-a934d874f3d5`, run `01a01692-884e-7ef3-9498-948dca7a25ed`
- trace `01a01763-0b5c-7252-ac4d-820ad2eccfeb`, run `01a01763-1a16-7d70-98d7-2c5064d39ee0`

The sibling blocks in the same prompt (`<memory_structure>`,
`<memory_management_in_/memories/AGENTS.md>`, `<memory_reminder>`,
`<skills_management_in_/memories/skills/>`) *are* interpolated and correctly render the
literal `/memories/` prefix, so the prompt states two contradictory path conventions for
memory and `tools.json` writes. The unrendered occurrences include the
"Always use the '{memory_path}' prefix" rule and two worked `edit_file` examples the
model is told to imitate verbatim, so the latent risk is the model copying a literal
placeholder path into a tool argument. No sampled tool call did so yet.

## Where to fix it

In the Fleet prompt-assembly code, find the block by searching for the literal strings
`additional_important_instructions_to_remember`, `{memory_path}AGENTS.md`, and
`{memory_path}tools.json`. That block is appended to the system prompt without the
`.format(memory_path=...)` / template render its siblings go through. Apply the same
interpolation mechanism the sibling blocks use so that:

- `{memory_path}AGENTS.md` renders as `/memories/AGENTS.md`
- `{memory_path}tools.json` renders as `/memories/tools.json`
- the bare prefix rule renders as `Always use the '/memories/' prefix`
- both `edit_file` worked examples render real `/memories/...` paths

If `memory_path` is not configurable anywhere, inline the literal `/memories/` prefix
instead of introducing a new variable. Then audit the whole prompt-assembly path for
other blocks whose source contains `{identifier}` but which skip the interpolation step,
and add a test that renders the FULL system prompt for a representative agent config and
asserts no unsubstituted `{identifier}` placeholder remains (allowlisting only the
intentional JSON literal `{"tools": [], "interrupt_config": {}}`). Do not change the
semantics of the memory instructions themselves.

## Why nothing was patched here

`chat-langchain` has no `memory_path` variable and none of this memory-management prompt
text. Its only Prompt Hub prompts are the docs-agent and guardrails prompts
(`public-chat-langchain-test`, `public-chat-langchain-guardrails-test`), neither of which
contains the affected block, so the fix cannot be made from this repository.
