"""Constants used throughout the backend package."""

import os

# The ``WEAVIATE_INDEX_NAME`` environment variable must be set so that
# deployments explicitly control which Weaviate index is used.  Falling back
# to a default value can easily lead to the wrong data being queried, so we
# raise an error if the variable is missing.
try:
    WEAVIATE_DOCS_INDEX_NAME = os.environ["WEAVIATE_INDEX_NAME"]
except KeyError as exc:
    raise RuntimeError("WEAVIATE_INDEX_NAME environment variable must be set") from exc
