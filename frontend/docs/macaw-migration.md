# Chat LangChain Macaw migration

This integration consolidates FEP-599 through FEP-605 into one draft targeting
`master`, as requested for a single review and demo. It includes the foundations
from #1232; the earlier per-ticket release-branch plan is superseded by this draft.
Merge and deployment remain review gates.

## Components and boundaries

The app pins published `@langchain/macaw-components@1.7.1`. Every active UI surface
uses its public components, icons, and tokens. Icons Macaw does not export
(microphone, feedback, trash) come from `@phosphor-icons/react`, pinned to
the same version Macaw depends on:

| Surface | Macaw integration |
| --- | --- |
| History and navigation | ChatConversationList, groups/items, Input, Dialog, menus and buttons |
| Welcome and follow-up input | One app-owned MessageComposer using ChatComposer, PlainInput, Toolbar, Action, Attachments |
| Transcript | ChatConversation, ChatMessage, message actions, Attachment, CodeLite, Link |
| Edit-and-rerun | ChatRichContent form with Macaw Textarea |
| Agent execution | ChatState, ChatToolCall, groups and reasoning disclosures |
| Account and help | Dialog, Select, Input, Avatar, DropdownMenu, Banner, Kbd, Button |

The application retains stream transport, per-owner credentials, history storage,
feedback, analytics, file serialization, and browser speech recognition. Thin
presentation adapters map those states to Macaw. Markdown/GFM remains app-owned
inside ChatMessage custom blocks; Macaw CodeLite renders fenced code with copy.
The native hidden file input bridges the browser file picker to the upload hook.
Macaw's plain-text composer input preserves pasted code and whitespace without
serializing prompts through a rich Markdown editor.
Raw tools remain gated by the existing `showToolCalls=false` product setting.

Legacy local UI primitives, progress animations, unmounted time-travel UI, and
unused UI dependencies have been removed. There is one public stylesheet import.
The public Tailwind preset aligns app utilities with Macaw tokens; the app's
utilities follow package utilities so responsive layout overrides work. Markdown
styles apply only to content and do not restyle CodeLite's internal tables.
The existing dark-only theme and packaged Inter/Fira Code fonts are retained.

## Run locally

Use **Node 26 or newer**. Chat LangChain remains an npm consumer with its existing
package-lock; Macaw's Nx/pnpm tooling belongs to the design-system repository.

```sh
cd frontend
npm ci
npm run dev:demo
```

Open `http://localhost:3000/dev/macaw`. This opt-in interactive demo uses the same
shell, composer, transcript, and dialogs as the app. Sample responses, tool rows,
and history stay in memory; the visible banner distinguishes them from live AI.
It supports local sending/queueing/stopping, attachments, editing, regeneration,
search, and mobile navigation. It never sends prompts to a model or tools.
Normal root auth/analytics providers still use any configured services.
The dynamic route returns 404 unless server-only `MACAW_SMOKE_TEST=1` is set.
Do not enable that flag in production.

For real chat, run `npm run dev` with the existing environment configuration and
open `/`. Use the Vercel draft preview for backend/auth validation when local
service configuration is unavailable. Secrets remain outside the repository.

## Verification and review gates

`npm test`, `npm run typecheck`, and `npm run build` cover the guest/auth security
contracts and the migration-sensitive adapters. Additional regressions protect
explicit tool completion/failure/interruption, stable tool IDs, restored history
and attachments, UTF-8 text files, prompt budget admission, queued-draft retention,
and duplicate URL-prompt prevention. Frontend CI runs a clean Node 26 install,
tests, and the default production build.

Browser review covers desktop and narrow-screen navigation, search/no-results,
new chat, shared composer, multiline entry, queued responses, stop/interruption,
Markdown/code, editing, disclosures, dialogs, and scroll behavior. Local fixture
checks establish presentation behavior, not live service correctness.

Before merging/deploying, validate the configured preview's real guest and signed-in
chat, merged owner histories, deep links/reload, region-specific OAuth/email paths,
feedback/trace metadata, voice permission behavior, and backend cancellation.
Confirm deployment runtime support for Node 26. FEP-605 remains open until the
approved PR is merged and representative production flows have been checked.

Rollback: revert the final integration commit on `master` and redeploy the prior
frontend through the normal Vercel process. This migration changes no backend
schemas, auth protocol, or persisted thread identifiers.
