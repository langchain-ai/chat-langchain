"""Constants used throughout the backend package."""

import os

# Allow overriding the default index name via the ``WEAVIATE_INDEX_NAME``
# environment variable so deployments can configure their own index without
# modifying the code.
WEAVIATE_DOCS_INDEX_NAME = os.environ.get(
    "WEAVIATE_INDEX_NAME", "LangChain_Combined_Docs_OpenAI_text_embedding_3_small"
)
