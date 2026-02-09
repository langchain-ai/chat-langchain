"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

MODEL_NAME_TO_RESPONSE_MODEL = {
    "anthropic_claude_3_5_sonnet": "anthropic/claude-3-5-sonnet-20240620",
}


def _update_configurable_for_backwards_compatibility(
    configurable: dict[str, Any],
) -> dict[str, Any]:
    update = {}
    if "k" in configurable:
        update["search_kwargs"] = {"k": configurable["k"]}

    if "model_name" in configurable:
        update["response_model"] = MODEL_NAME_TO_RESPONSE_MODEL.get(
            configurable["model_name"], configurable["model_name"]
        )

    if update:
        return {**configurable, **update}

    return configurable


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["weaviate"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="weaviate",
        metadata={"description": "The vector store provider to use for retrieval."},
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    # for backwards compatibility
    k: int = field(
        default=6,
        metadata={
            "description": "The number of documents to retrieve. Use search_kwargs instead."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        configurable = _update_configurable_for_backwards_compatibility(configurable)
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)
