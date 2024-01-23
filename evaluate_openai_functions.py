
import os
import argparse

from langchain.smith import RunEvalConfig
from langsmith import Client

from croptalk.model_openai_functions import model

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')

if __name__ == "__main__":

    model_name = os.getenv("MODEL_NAME")
    test_dataset = os.getenv("TEST_DATASET")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--dataset-name", default=test_dataset)
    args = parser.parse_args()

    # Load default dataset
    client = Client()

    # https://api.python.langchain.com/en/latest/smith/langchain.smith.evaluation.config.RunEvalConfig.html
    eval_config = RunEvalConfig(
        evaluators=["qa"],
        reference_key="Answer",  # key in the dataset to compare against
        input_key="question",  # traced run’s inputs dictionary
        # prediction_key="llm_output",  # traced run: outputs dict
    )

    results = client.run_on_dataset(
        dataset_name=test_dataset,
        llm_or_chain_factory=model,
        evaluation=eval_config,
        input_mapper=lambda x: {
            "question": f"Context : {x['Context']}.\n Question: {x['Question']}"},
        # project_name="CropTalk Knowledge", # created automatically
        tags=["test_context", "simple_chain"],
        verbose=True,
    )

    print(results)
