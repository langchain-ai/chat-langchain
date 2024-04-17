import ast
import logging
from typing import Tuple, Dict

import pandas as pd
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer

from _scripts.utils import get_nodes, parse_args, get_output_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate_arguments(df):
    equal_arguments = []
    for i, j in zip(list(df["expected_arguments"]), list(df["actual_arguments"])):
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
    logger.info(f"Evaluating croptalk's tools capacity, using config: {args}\n")

    if args.use_model_llm:
        from croptalk.model_llm import model
    else:
        from croptalk.model_openai_functions import model

    if args.eval_path:
        # getting eval df
        eval_df = pd.read_csv(args.eval_path, sep=";")

        # running each scenario
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

        # adding data to df
        eval_df["actual_arguments"] = input_actual
        eval_df["actual_output"] = output_actual
        eval_df["expected_arguments"] = eval_df["expected_arguments"].apply(ast.literal_eval)
        eval_df["actual_arguments"] = eval_df["actual_arguments"].apply(ast.literal_eval)

        # evaluating output and arguments
        eval_df = evaluate_arguments(eval_df)
        eval_df = evaluate_output(eval_df)

        # saving to csv
        output_path = get_output_path(args.eval_path, args.use_model_llm)
        eval_df.to_csv(output_path)

    else:
        with tracing_v2_enabled() as langchain_tracer:
            # model.invoke({
            #     "chat_history": [],
            #     "question": "find me the SP document for Whole Farm in yakima county washington for 2024"
            # })

            # model.invoke({
            #     "chat_history": [],
            #     "question": "find me the SP document for Whole Farm Revenue in yakima county washington"
            # })

            # model.invoke({
            #     "chat_history": [],
            #     "question": "SP document for Corn, in Butte County, California"
            # })

            model.invoke({
                "chat_history": [],
                "question": "SP document apples in yakima county in washington"
            })
