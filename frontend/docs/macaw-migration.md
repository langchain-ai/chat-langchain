# Chat LangChain Macaw migration

This document defines the shared integration branch and the application boundaries
for [FEP-599](https://linear.app/langchain/issue/FEP-599). It is the starting point
for the independently reviewed migration slices, not a claim that the complete
Macaw chat family is ready to ship.

## Branch workflow

- Shared branch: `codex/macaw-migration-release` in
  [`langchain-ai/chat-langchain`](https://github.com/langchain-ai/chat-langchain).
- Starting point: `master` at `fd18b87`, the default-branch head checked on
  November 4, 2026.
- Create each ticket's feature branch from the current shared branch and set its
  PR base to `codex/macaw-migration-release`. Merge reviewed migration slices into
  that branch.
- [FEP-605](https://linear.app/langchain/issue/FEP-605) owns the final integration
  review, removal of superseded UI, and a squash merge of the completed release
  branch into the current default branch. That branch is currently `master`;
  recheck its name and latest changes before the final PR. Deployment and
  production smoke checks follow that final merge.

## Selected package and release gate

The foundations work pins the published `@langchain/macaw-components` package to
**`1.6.0`**. Use public package entry points, including component subpaths, rather
than copying component implementations or importing package internals. Shared
component fixes belong in the Macaw repository.

Readiness checked on November 4, 2026:

| Component | Availability in the selected release |
| --- | --- |
| `ChatComposer` | Published in `1.6.0` |
| `ChatMessage` | Published in `1.6.0` |
| `ChatConversationList` | Published in `1.6.0` |
| `ChatToolCall` | Published in `1.6.0` |
| `ChatState` | Published in `1.6.0` |
| `ChatConversation` | Not available; [Macaw #77](https://github.com/langchain-ai/macaw-design-system/pull/77) was open |
| `ChatRichContent` | Not available; [Macaw #88](https://github.com/langchain-ai/macaw-design-system/pull/88) was closed without merging |

The shared branch and the available `1.6.0` foundations are integrated: dependency,
theme provider, package styles/fonts, legacy CSS coexistence, and a local smoke
fixture. Migration slices using the published components can build on this setup.
**FEP-599's completed-chat-family acceptance criterion still requires a future
published release containing the missing components.** Recheck the registry
artifact and exports when upgrading; a merged source PR alone does not establish
package availability. Update this table and the exact package pin together.

`ChatRichContent` supplies interactive forms and approval cards. The current app
does not have those features. Its availability is a package-readiness requirement,
not an instruction to add new product features or replace Markdown rendering.

## Runtime and consumer setup

- Use Node **26 or newer** for the selected Macaw release. `package.json` declares
  that minimum and `frontend/.nvmrc` selects Node 26; use it for local development,
  CI, and the eventual frontend deployment.
- The app retains Next **16.4.3** and React/React DOM **19.1.0**. The React type
  packages are aligned at **19.3.0**; component ref declarations account for their
  nullable values.
- Tailwind and `@tailwindcss/postcss` are pinned to **4.3.4**, satisfying Macaw's
  `^4.3.3` peer range.
- This repository's frontend uses **npm** and `frontend/package-lock.json`.
  Preserve that lockfile and use `npm ci` for a clean install. Macaw's own Nx/pnpm
  workspace does not change this consumer's package manager.
- `app/layout.tsx` remains a server component. The application-owned
  `components/providers/macaw-provider.tsx` client boundary wraps
  `AppThemeProvider` and the public `TooltipProvider`. `app/globals.css` imports the public
  `@langchain/macaw-components/styles.css` stylesheet exactly once.

### Theme, fonts, and legacy styles

The current application keeps its fixed dark palette. `MacawProvider` sets
`defaultMode="dark"` and `storageKey={null}`, so saved or system preferences do not
switch the app's theme. The server-rendered `<html>` supplies matching `dark`
class, theme data attributes, and color scheme. That document-wide theme also
covers dialogs and tooltips rendered into `document.body`.

The package stylesheet supplies Inter and Fira Code. The root layout no longer
downloads Inter/Inconsolata through `next/font/google`; global font aliases now
use the packaged Inter/Fira Code fonts. The change to Fira Code for monospaced
content is intentional.

Legacy semantic colors use `--chat-*` variables and `*-chat-*` Tailwind utilities,
including `bg-chat-background` and `text-chat-foreground`. This prevents Macaw's
generic tokens and `html.dark` declarations from changing the existing palette.
The stylesheet declares cascade-layer order explicitly, with Macaw's compiled
utilities after the app's utility layer. The legacy page surface and Markdown
styles remain application-owned.

Legacy small text uses `text-chat-xs` to retain its 12px size and 16px line height
without overriding Macaw's `text-xs` utility. `lib/utils/index.ts` registers
`text-chat-xs` as a font-size class with `tailwind-merge`, keeping size overrides
distinct from text-color utilities. Legacy input and textarea controls receive a
visible keyboard-focus outline because Macaw's base styles remove their focus
box shadows.

### Local smoke fixture

`app/dev/macaw/page.tsx` exposes `/dev/macaw` only when the server process has
`MACAW_SMOKE_TEST=1`. The route is dynamic, returns 404 by default, and excludes
itself from indexing. For a production-build check, run from `frontend/`:

```sh
NEXT_PUBLIC_LANGGRAPH_API_URL=http://127.0.0.1:2024 npm run build
MACAW_SMOKE_TEST=1 NEXT_PUBLIC_LANGGRAPH_API_URL=http://127.0.0.1:2024 npm start
```

Open `/dev/macaw` and check the Macaw button, message state, tooltip portal, fonts,
and legacy control focus. Then restart without the flag and confirm the route
returns 404. The flag is server-only and is intended for local verification.

The fixture component uses local display state and has no chat backend callbacks.
The normal root authentication and analytics providers still wrap it and may
contact their configured services. Fixture results do not establish live chat or
authenticated product behavior.

### Continuous integration

`.github/workflows/frontend.yml` runs for frontend changes in PRs targeting
`master` or `codex/macaw-migration-release`, and for pushes to the release branch.
It selects Node 26 from `.nvmrc`, enables Socket's package firewall, installs the
committed lockfile with `npm ci`, and runs `npm test` and `npm run build`. The
workflow uses read-only repository permissions and pinned action revisions.
It supplies a loopback backend URL to satisfy build-time configuration; it does
not connect to a deployed chat service.

## Application-owned composition boundaries

Adapters should translate existing app state into component props at the render
boundary. They must not become a second data store or require a new shared chat
domain model. Keep the following contracts when each slice is implemented.

| Surface | Existing source and thin composition contract |
| --- | --- |
| Navigation and history | `app/page.tsx`, `components/layout/sidebar.tsx`, and `lib/hooks/threads/use-threads.ts` own threads. Preserve `thread.thread_id` as the item ID and selection value; derive labels from `metadata.title`, previews from `metadata.lastMessage`, and dates from `updated_at`/`created_at`. Pass `onSelectThread(threadId)` and `onDeleteThread(threadId)` through to the existing app handlers. Filtering, grouping, and empty/loading display can be composed around `ChatConversationList`. |
| Composer | `components/chat/chat-interface.tsx`, `components/chat/chat-input.tsx`, and `lib/hooks/chat/use-chat-state.ts` own the draft and submission lifecycle. Compose `ChatComposer` with the controlled input value/change callback and existing send/stop callbacks. Preserve length validation, keyboard and paste handling, attachments, voice input, focus refs, and queued-message behavior. The app remains responsible for draft persistence, uploads, request cancellation, and queue processing. |
| Transcript and message bodies | `lib/types/messages.ts` defines `Message`; `components/chat/message-list.tsx` and `message-item.tsx` render it. Preserve `Message.id` and its `user`/`assistant` role. Keep the existing Markdown renderer and attachment rendering as application-owned body children. Compose those bodies with `ChatMessage` and, once published, `ChatConversation`. The conversation container handles presentation and scroll behavior; fetching history and applying streamed updates stay in the app. |
| Message actions | Preserve the existing callbacks: `onCopy(content, messageId)`, `onEditAndRerun(messageId, newContent)`, `onRegenerate()`, and `onFeedback(messageId, feedbackType, comment?)`. Pass presentation state such as the copied ID, feedback value, and regeneration availability. Keep clipboard writes, rerun requests, feedback persistence, tracing, and analytics in the existing handlers. |
| Progress and execution details | `lib/hooks/chat/use-stream-handler.ts`, `lib/types/tools.ts`, and the existing message fields own execution data. Map tool `id`, `name`, `args`, and `output` into `ChatToolCall` display props; map explicit thinking/streaming/completion/error state into `ChatState`. Preserve subgraph `toolCallId`, `isStreaming`, and `isComplete` where supplied. Missing tool output alone is not proof of success or completion. |
| Account and supporting dialogs | `components/layout/header.tsx`, `components/auth/AuthModal.tsx`, and `lib/auth/AuthProvider.tsx` own authentication. Use Macaw primitives for the avatar, menu, controls, and dialog composition. Preserve controlled open/change callbacks, provider/email sign-in callbacks, sign-out, region selection, and current loading/error states. |

Thread-owner selection, guest identity and stored thread IDs, authorization,
streaming, backend persistence, LangSmith operations, Segment events, and tool
execution remain in Chat LangChain. Do not pass credentials into presentational
adapters. Do not regenerate thread or message IDs during rendering. Preserve the
existing lifecycle distinctions (`isLoadingThread`, `isLoading`, `isRegenerating`,
`isStopping`, message thinking state, interruption, and errors) rather than
collapsing all states into a single busy flag.

## Incremental implementation order

1. **FEP-599 — foundations and dependencies:** establish the shared branch,
   published package pin, consumer runtime, theme/style boundary, and this guide.
   Complete the release gate and foundation validation before declaring the
   ticket accepted.
2. **FEP-600 — navigation and thread history:** integrate the shell and
   conversation list while preserving thread ownership and navigation handlers.
3. **FEP-601 — composer:** replace composer presentation while preserving
   submission, attachments, voice, queue, and stop behavior.
4. **FEP-602 — transcript, message content, and actions:** integrate conversation
   and message presentation, retaining app-owned Markdown and action handlers.
5. **FEP-603 — agent progress and execution details:** compose tool calls and
   execution states from existing stream data.
6. **FEP-604 — account, authentication, and supporting dialogs:** replace their
   presentation without changing authentication or analytics behavior.
7. **FEP-605 — integrated validation and final merge:** validate all slices
   together, remove replaced legacy UI/dependencies, review the final diff, squash
   merge to the current default branch, deploy, and perform production smoke
   checks.

Independent surfaces using the available components can proceed in parallel on
the foundations branch using these contracts. Work requiring `ChatConversation`
must wait for its published release. Revalidate the combined release branch as
slices land, and satisfy the complete-family gate before the final integration.

## Validation record

Results for the `1.6.0` foundations change on November 4, 2026:

- [x] Clean install from the committed lockfile using Node 26: `npm ci`.
- [x] Dependency audit during clean installation: **0 vulnerabilities**.
- [x] Existing guest/auth tests after dependency changes: **35/35** passed.
- [x] TypeScript validation passed after updating React ref types.
- [x] Production frontend build using `npm run build -- --webpack`, with the
  loopback backend URL above. The default Turbopack builder hit a local process
  port permission error (`EPERM`); the CI workflow checks that default builder.
- [x] Browser smoke of the production build: Button, ChatMessage, ChatState and
  Tooltip render and respond to interaction. A fresh reload produced no console
  errors or hydration warnings.
- [x] Document theme and body-mounted tooltip remain dark; packaged fonts load;
  the legacy primary color remains `#7FC8FF`, small text remains 12px/16px, and
  keyboard focus on the legacy textarea has a visible 2px outline.
- [x] `/dev/macaw` returns 404 without the server flag and 200 with it enabled.
- [ ] Existing guest and authenticated chat flows remain usable: thread
  selection, sending/streaming/stopping, reload/history, and sign-in/sign-out.
  The local app renders its authentication UI, but no live auth/backend
  configuration was supplied for these end-to-end checks.

Record any missing local configuration or external service access rather than
marking those flows verified. Component fixtures can establish styling,
interaction, and hydration behavior, but they do not prove live backend streaming,
authorization, persistence, or authenticated product behavior.
