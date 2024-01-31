# TODO: Consolidate all these scripts into a single script
# This is ugly
import argparse

from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.smith import RunEvalConfig
from langsmith import Client

# Ugly. Requires PYTHONATH=$(PWD) to run
from main import create_chain, get_retriever

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
}

_MODEL_MAP = {
    "openai": "gpt-3.5-turbo-1106",
    "anthropic": "claude-2",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Complex Questions")
    parser.add_argument("--model-provider", default="openai")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)
    retriever = get_retriever()
    llm = _PROVIDER_MAP[args.model_provider](
        model=_MODEL_MAP[args.model_provider], temperature=0
    )

    # In app, we always pass in a chat history, but for evaluation we don't
    # necessarily do that. Add that handling here.
    def construct_eval_chain():
        chain = create_chain(
            retriever=retriever,
            llm=llm,
        )
        return {
            "question": lambda x: x["question"],
            "chat_history": (lambda x: x.get("chat_history", [])),
        } | chain

    eval_config = RunEvalConfig(
        evaluators=["qa"],
        prediction_key="output",
    )
    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        llm_or_chain_factory=construct_eval_chain,
        evaluation=eval_config,
        tags=["simple_chain"],
        verbose=True,
    )
