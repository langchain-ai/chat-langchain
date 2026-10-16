"""Extend managed authentication with the public chat authorization policy."""

from typing import cast

from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user
from managed_deepagents.runtime import build_managed_auth

from identity import identity

auth = cast(Auth, build_managed_auth(identity))


@auth.on.crons
async def authorize_crons(ctx: Auth.types.AuthContext, value: dict) -> None:
    """Allow only Studio actors to manage scheduled runs."""
    if not is_studio_user(ctx.user):
        raise Auth.exceptions.HTTPException(
            status_code=403, detail="Cron jobs are restricted to Studio users"
        )
