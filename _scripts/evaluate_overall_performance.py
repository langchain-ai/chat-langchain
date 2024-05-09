import os

os.chdir('/app')

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI

from croptalk.model_openai_functions import initialize_model
from dotenv import load_dotenv

load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')

# Define the dataset name and output directory
DATASET_NAME = "sp-cp-bp-cih-sql-89q"
OUTPUT_DIR = "data/experiment_results"

# Initialize a model that will be used for evaluation of the qa
eval_llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=False, temperature=0.0)

client = Client()

# == CORRECTNESS EVALUATOR ==
def retrieve_question_answer(run: Run, example: Example) -> dict:
    parsed = {
        "prediction": run.outputs.get("output"),
        "reference": example.outputs.get("Answer"),
        "input": example.inputs.get("Question"),
    }
    return parsed

qa_evaluator = LangChainStringEvaluator(
    "qa",
    prepare_data=retrieve_question_answer,
    config={"llm": eval_llm},
)

# == DOC RETRIEVAL EVALUATOR ==
def doc_evaluator(run: Run, example: Example) -> dict:
    """Calculate the exact match score of the run."""
    expected = example.outputs.get("Doc Name")
    predicted = run.outputs.get("output")
    return {"score": expected.lower() in predicted.lower() if expected else None,
            "key": "contains_doc"}

# == TOOL EVALUATOR ==
def tool_evaluator(run: Run, example: Example) -> dict:
    
    # Accessing the `intermediate_steps`
    #TODO: Parse the intermediate steps into a list of objects
    # For now, we just convert the intermediate steps to a string
    intermediate_steps = run.outputs.get('intermediate_steps')
    intermediate_steps_str = str(intermediate_steps)

    expected_tool_name = example.outputs.get("Tool")
    expected_tool = f"tool='{expected_tool_name}'"
    return {"score": expected_tool.lower() in intermediate_steps_str.lower() if expected_tool_name else None,
            "key": "expected_tool"}


# Helper to run the model
MODEL, _ = initialize_model(convert_response_chain_to_str=False, no_memory=True)

def predict(inputs: dict) -> dict:
    print(f"inputs: {inputs}")
    return MODEL.invoke({"question": inputs["Question"], 
                         "chat_history": []})

# Evaluate the model
experiment_results = evaluate(
    predict, 
    data=DATASET_NAME, # The data to predict and grade over
    evaluators=[qa_evaluator, doc_evaluator, tool_evaluator], # The evaluators to score the results
    experiment_prefix="april", # A prefix for your experiment names to easily identify them
)

# Save the results to a CSV file
experiment_name = experiment_results.experiment_name

filename = os.path.join(OUTPUT_DIR, f"{experiment_name}.csv")
client.get_test_results(project_name=experiment_name).to_csv(filename, index=False)

print(f"Experiment results saved to {filename}")