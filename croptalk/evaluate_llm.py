import argparse

from langchain.smith import RunEvalConfig
from langsmith import Client

from croptalk.model_agent import initialize_llm

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')


def main(model_name, dataset_name):
    llm = initialize_llm(model_name)

    # Load default dataset
    client = Client()

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
        dataset_name=dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        input_mapper=lambda x: x["Question"],
        project_name="CropTalk Knowledge",
        tags=["simple_chain"],
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of the model",
                        default="gpt-3.5-turbo-1106")
    parser.add_argument(
        "--dataset", help="Name of the dataset", default="Default Dataset")
    args = parser.parse_args()

    main(args.model, args.dataset)
