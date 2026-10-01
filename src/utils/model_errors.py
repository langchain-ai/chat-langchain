"""Helpers for classifying provider model errors."""


def is_auth_error(exc: BaseException) -> bool:
    """Return whether an exception indicates invalid model credentials."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).upper()
        if any(
            marker in message
            for marker in (
                "API_KEY_INVALID",
                "PERMISSION_DENIED",
                "API KEY NOT VALID",
            )
        ):
            return True

        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code in (401, 403):
            return True

        current = current.__cause__ or current.__context__

    return False


__all__ = ["is_auth_error"]
