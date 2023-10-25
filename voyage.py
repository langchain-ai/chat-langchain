from typing import List, Optional

import requests
import json

from langchain.schema.embeddings import Embeddings
from langchain.utils.iter import batch_iterate
from langchain.utils import get_from_env

BATCH_SIZE = 6


class VoyageEmbeddings(Embeddings):
    """Voyage AI embedding model wrapper."""

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.url = url or get_from_env("url", "VOYAGE_AI_URL")
        self.model = model or get_from_env("model", "VOYAGE_AI_MODEL")
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for batch in batch_iterate(self.batch_size, texts):
            data = json.dumps({"input": batch, "model": self.model})
            response = requests.post(
                self.url, headers={"Content-Type": "application/json"}, data=data
            )
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"Received status code {response.status_code} and response "
                    f"{response.text}"
                )
            response_data = response.json()["data"]
            embeddings.extend([x["embedding"] for x in response_data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
