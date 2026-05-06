# Standardized trace metadata for LangSmith observability
from typing import Literal

# Source types for trace metadata
# - Chat-LangChain: Public chat interface (chat.langchain.com)
# - Internal-Chat-LangChain: Internal chat interface (jewel deployment, auto-set
#   by @auth.on.threads.create_run when LANGSMITH_HOST_PROJECT_NAME="jewel")
# - Support-Portal: Public help portal (help-portal website)
# - Slack: Slack bot integration
# - Pylon: Pylon ticket system (use pylon_channel="chat"|"email" to distinguish)
# - Studio: LangGraph Studio
SourceType = Literal[
    "Chat-LangChain",
    "Internal-Chat-LangChain",
    "Support-Portal",
    "Slack",
    "Pylon",
    "Studio",
]


def build_trace_metadata(
    *,  # Force keyword arguments for clarity
    # User context
    user_id: str,
    user_email: str | None = None,
    user_name: str | None = None,
    user_org: str | None = None,
    # Source context
    source_type: SourceType = "Chat-LangChain",
    channel_id: str | None = None,
    # Ticket context (Pylon)
    ticket_id: str | None = None,
    ticket_number: str | int | None = None,
    ticket_priority: str | None = None,
    ticket_category: str | None = None,
    ticket_status: str | None = None,
    # Agent context
    graph_id: str | None = None,
    # Prompt provenance - "hub:docs-agent-prompt:production" or "local:src/prompts/..."
    prompt_source: str | None = None,
    # Prompt Hub commit hash (None when using local fallback)
    prompt_commit: str | None = None,
    # Additional fields
    **extra,
) -> dict:
    metadata = {"user_id": user_id, "source_type": source_type}

    # Optional fields - only add if present
    optional = {
        "user_email": user_email,
        "user_name": user_name,
        "user_org": user_org,
        "channel_id": channel_id,
        "ticket_id": ticket_id,
        "ticket_number": str(ticket_number) if ticket_number is not None else None,
        "ticket_priority": ticket_priority,
        "ticket_category": ticket_category,
        "ticket_status": ticket_status,
        "graph_id": graph_id,
        "prompt_source": prompt_source,
        "prompt_commit": prompt_commit,
    }

    for key, value in optional.items():
        if value is not None:
            metadata[key] = value

    metadata.update(extra)
    return metadata


# Common email providers that don't indicate an organization
_PERSONAL_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "yahoo.co.uk",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "icloud.com",
        "me.com",
        "mac.com",
        "protonmail.com",
        "proton.me",
        "aol.com",
        "mail.com",
    }
)


def extract_org_from_email(email: str) -> str | None:
    if not email or "@" not in email:
        return None

    domain = email.split("@")[1].lower()

    if domain in _PERSONAL_EMAIL_DOMAINS:
        return None

    # Extract org name from domain
    parts = domain.split(".")
    if len(parts) < 2:
        return parts[0]

    # Handle multi-part TLDs like .co.uk
    if parts[-2] in {"co", "com", "org", "net", "io", "ai", "dev"}:
        return parts[-3] if len(parts) >= 3 else parts[0]

    return parts[-2]
