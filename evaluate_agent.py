
import os
import argparse
import time

from langchain.smith import RunEvalConfig
from langsmith import Client

from croptalk.model_agent import initialize_agent_executor
from croptalk.tools import tools

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":

    model_name = os.getenv("MODEL_NAME")
    test_dataset = os.getenv("TEST_DATASET")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--dataset-name", default=test_dataset)
    args = parser.parse_args()

    client = Client()
    agent_executor = initialize_agent_executor(
        model=args.model_name, tools=tools)

    eval_config = RunEvalConfig(
        evaluators=["qa"],
        # The key from the traced run’s outputs dictionary to use to represent the prediction.
        prediction_key="output",
        # The key from the traced run’s inputs dictionary to use to represent the input.
        input_key="Question",
        output_key="result",
        # The key in the dataset run to use as the reference string.
        reference_key="Answer",
    )

    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        llm_or_chain_factory=agent_executor,
        evaluation=eval_config,
        input_mapper=lambda x: {"question": x["Question"]},
        project_name=f"CropTalk Knowledge {time.time()}",
        tags=["agent"],
        concurrency_level=0,  # Add this to not go async
        verbose=False,
    )

    print(results)
