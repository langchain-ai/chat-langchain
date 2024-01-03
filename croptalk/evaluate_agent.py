
import argparse
import time

from langchain.smith import RunEvalConfig
from langsmith import Client

from croptalk.model import initialize_agent_executor
from croptalk.tools import tools

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt-3.5-turbo-1106")
    parser.add_argument("--dataset-name", default="Default Dataset")
    args = parser.parse_args()

    client = Client()
    agent_executor = initialize_agent_executor(model=args.model_name, tools=tools)

    eval_config = RunEvalConfig(
            evaluators=["qa"],
            prediction_key="output", # The key from the traced run’s outputs dictionary to use to represent the prediction.
            input_key="Question", # The key from the traced run’s inputs dictionary to use to represent the input. 
            output_key="result",
            reference_key="Answer", # The key in the dataset run to use as the reference string.
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