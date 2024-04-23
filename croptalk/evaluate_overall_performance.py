import os

os.chdir('/app')

from langchain.smith import RunEvalConfig
from langsmith import Client

from croptalk.model_openai_functions import model as MODEL

from dotenv import load_dotenv

load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')

DATASET_NAME = "sp-cp-bp-cih-70"

# DOCS: 
# https://api.python.langchain.com/en/latest/smith/langchain.smith.evaluation.config.RunEvalConfig.html

eval_config = RunEvalConfig(
    evaluators=["qa"],
    reference_key="Answer",  # key in the dataset to compare against
    input_key="question",  # traced run’s inputs dictionary
)

client = Client()
results = client.run_on_dataset(
    dataset_name=DATASET_NAME,
    llm_or_chain_factory=MODEL,
    evaluation=eval_config,
    input_mapper=lambda x: {"question": x["Question"], 
                            "chat_history": []},
    # project_name="CropTalk AprDemo", # has to be unique
    # tags=["simple_chain"],           # list of tags 
    verbose=True,
)