"""Managed LangSmith connector for Chat LangChain.

Auto-discovered by the compiler (never imported by ``agent.py``). Replaces the
hand-rolled ``src/api/langsmith_routes.py`` browser proxy: the browser calls
``POST /connectors/langsmith/capabilities/{id}`` with its own identity token, and
MDA performs the LangSmith operation server-side with the workspace key
(``LANGSMITH_API_KEY``) and returns an allowlisted response. The browser never
receives the LangSmith key.

Capabilities mirror exactly what the old routes exposed:

- ``langsmith:chat-feedback`` — thumbs up/down feedback on a run (create / update
  / delete), one per actor, bounded comment.
- ``langsmith:trace-viewer`` — read a redacted run summary and create a public
  share link (``runs.read`` / ``runs.share``).
"""

from managed_deepagents.connectors import langsmith

connector = langsmith.connector(
    langsmith.feedback(
        id="langsmith:chat-feedback",
        expose_to=["browser"],
        actions=["create", "update", "delete"],
        scope="run",
        keys=["ux.thumb_vote"],
        scores=["1", "0"],
        max_comment_chars=2000,
        one_per_actor=True,
    ),
    # trace_viewer() with an extended include so the browser can still render
    # token usage + cost + timing on each message (the default preset omits them).
    langsmith.runs(
        id="langsmith:trace-viewer",
        expose_to=["browser"],
        actions=["read", "share"],
        scope="thread",
        include=[
            "id",
            "status",
            "start_time",
            "end_time",
            "url",
            "metadata",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cost",
            "completion_cost",
            "total_cost",
        ],
    ),
)
