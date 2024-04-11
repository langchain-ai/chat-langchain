import logging
from typing import Tuple, Dict

from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer

from _scripts.utils import get_nodes, parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate_arguments(df):
    equal_arguments = []
    for i, j in zip(list(df["expected_arguments"]), list(df["actual_arguments"])):
        print(i, type(i))
        print(j, type(j))
        equal_arguments.append(i == j)

    df["equal_arguments"] = equal_arguments
    return df


def evaluate_output(df):
    equal_outputs = []
    for i, j in zip(list(df["expected_output"]), list(df["actual_output"])):
        equal_outputs.append((str(i) in str(j)))

    df["equal_outputs"] = equal_outputs
    return df


def get_actual_results(langchain_tracer: LangChainTracer, name: str) -> Tuple[Dict, Dict]:
    input_result = get_nodes(
        root_node=langchain_tracer.latest_run,
        node_name=name,
    )[0].inputs["input"]

    output_result = get_nodes(
        root_node=langchain_tracer.latest_run,
        node_name=name,
    )[0].outputs["output"]

    return input_result, output_result


if __name__ == "__main__":
    # load model
    logger.info("Loading model")

    # parse args
    args = parse_args()
    logger.info(f"Evaluating croptalk's document retrieval capacity, using config: {args}\n")

    import pandas as pd
    import ast

    if args.use_model_llm:

        from croptalk.model_llm import model
    else:
        from croptalk.model_openai_functions import model

    eval_df = pd.read_csv("_scripts/evaluate_tools.csv", sep=";")

    eval_df["expected_arguments"] = eval_df["expected_arguments"].apply(ast.literal_eval)

    input_actual, output_actual = list(), list()
    for i, row in eval_df.iterrows():
        with tracing_v2_enabled() as langchain_tracer:
            model.invoke({
                "chat_history": [],
                "question": str(row["query"])
            })

        input_result, output_result = get_actual_results(langchain_tracer, str(row["tool_used"]))
        input_actual.append(input_result)
        output_actual.append(output_result)

    eval_df["actual_arguments"] = input_actual
    eval_df["actual_arguments"] = eval_df["actual_arguments"].apply(ast.literal_eval)

    eval_df["actual_output"] = output_actual

    eval_df = evaluate_arguments(eval_df)
    eval_df = evaluate_output(eval_df)
    eval_df.to_csv(f"_scripts/evaluation_{args.use_model_llm}.csv")
