import logging

import pandas as pd
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer

from typing import Tuple, Dict
from _scripts.utils import get_nodes, parse_args

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
    # print("LATEST RUN")
    # print(langchain_tracer.latest_run)

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
    from croptalk.model_llm import model
    import ast

    # parse args
    args = parse_args()
    logger.info(f"Evaluating croptalk's document retrieval capacity, using config: {args}\n")

    if args.use_model_llm:
        from croptalk.model_llm import model

        memory = None
    else:
        from croptalk.model_openai_functions import model, memory

    # eval_df = pd.read_csv("_scripts/evaluate_tool.csv")
    # eval_df["expected_arguments"] = eval_df["expected_arguments"].apply(ast.literal_eval)
    #
    # input_actual, output_actual = list(), list()
    # for i, row in eval_df.iterrows():
    #     with tracing_v2_enabled() as langchain_tracer:
    #         model.invoke({
    #             "chat_history": [],
    #             "question": str(row["query"])
    #         })
    #
    #     input_result, output_result = get_actual_results(langchain_tracer, str(row["tool_used"]))
    #     input_actual.append(input_result)
    #     output_actual.append(output_result)
    #
    # eval_df["actual_arguments"] = input_actual
    # eval_df["actual_output"] = output_actual
    #
    # eval_df = evaluate_arguments(eval_df)
    # eval_df = evaluate_output(eval_df)

    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the cost to grower under the APH policy for walnuts in Fresno county in California in 2023",
    # })


    # model.invoke({
    #     "chat_history": [],
    #     "question": "What is the distribution of policy sold amongst counties for "
    #                 "the WFRP policy for the state of Kansas",
    # })

    # model.invoke({
    #     "chat_history": [],
    #     "question": "How many WFRP policies were sold in NY in 2023",
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
