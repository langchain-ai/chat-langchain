import logging

import pandas as pd
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer

from typing import Tuple, Dict
from _scripts.utils import get_nodes

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


def get_actual_results(langchain_tracer: LangChainTracer) -> Tuple[Dict, Dict]:
    input_result = get_nodes(
        root_node=langchain_tracer.latest_run,
        node_name="ToolInput",
    )[0].outputs["arguments"]

    output_result = get_nodes(
        root_node=langchain_tracer.latest_run,
        node_name="ToolOutput",
    )[0].outputs["output"]

    return input_result, output_result


if __name__ == "__main__":
    # load model
    logger.info("Loading model")
    from croptalk.model_llm import model
    import ast

    eval_df = pd.read_csv("_scripts/evaluate_tool.csv")
    eval_df["expected_arguments"] = eval_df["expected_arguments"].apply(ast.literal_eval)

    input_actual, output_actual = list(), list()
    for i, row in eval_df.iterrows():
        with tracing_v2_enabled() as langchain_tracer:
            model.invoke({
                "chat_history": [],
                "question": str(row["query"])
            })

        input_result, output_result = get_actual_results(langchain_tracer)
        input_actual.append(input_result)
        output_actual.append(output_result)

    eval_df["actual_arguments"] = input_actual
    eval_df["actual_output"] = output_actual

    eval_df = evaluate_arguments(eval_df)
    eval_df = evaluate_output(eval_df)

    eval_df.to_csv("_scripts/tool_evaluation.csv")

    # model.invoke({
    #     "chat_history": [],
    #     "question": "What are the livestock commodities insured with WFRP for reinsurance year 2024,"
    #                 " state code 04 and county code 001 ",
    # })
    #
    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the percentage of policies indemnified for Washakie county in Wyoming "
    #                 "for oranges under the APH program"
    # })

    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the percentage of policies indemnified for Wyoming "
    #                 "under the APH program"
    # })

    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the number of policies sold for Bee county in Texas, for corn, for the RP program"
    # })

    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the number of policies sold for Bee county in Texas, for corn, "
    #                 "for the RP program, for 0.7 coverage level"
    # })

    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the number of policies sold for Bee county in Texas"
    # })
