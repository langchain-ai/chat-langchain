"""Helpers for classifying provider model errors."""


def is_auth_or_permission_error(error: BaseException) -> bool:
    """Return whether an error indicates invalid credentials or permissions."""
    if "API_KEY_INVALID" in str(error).upper():
        return True

    for candidate in (error, getattr(error, "response", None)):
        status_code = getattr(candidate, "status_code", None)
        if status_code in (401, 403):
            return True
    return False
