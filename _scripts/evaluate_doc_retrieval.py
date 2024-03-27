import argparse
from datetime import datetime
import logging
import os
from typing import List, NamedTuple

import json
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run
from langchain.schema.runnable import Runnable
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-model-llm",
        help="Option which, when specified, tells the evaluation to use model_llm (i.e. use model_openai_functions when this option is not specified)",
        action='store_true',
    )
    parser.add_argument(
        "eval_path",
        help="CSV file path that contains evaluation use cases",
    )
    return parser.parse_args()


def get_nodes(root_node: Run, node_name: str) -> List[Run]:
    """
    Args:
        root_node: tracing root node of a langchain call
        node_name: name of nodes we are looking for

    Returns:
        a list of tracing nodes found under provided root node's tree, whose name matches provided
        node name
    """
    res = []
    # recursively call this function on each child
    for child in root_node.child_runs:
        res.extend(get_nodes(child, node_name))
    # check if root node matches
    if root_node.name == node_name:
        res.append(root_node)
    return res


def find_last_finddocs_node(
    langchain_tracer: LangChainTracer, output_df: pd.DataFrame, eval_use_case: NamedTuple,
) -> Run:
    # retrieve all FindDocs nodes
    finddocs_nodes = get_nodes(
        root_node=langchain_tracer.latest_run,
        node_name="FindDocs",
    )
    output_df.loc[output_df.index == eval_use_case.Index, "nb_of_FindDocs_nodes_actual"] = len(finddocs_nodes)
    # return last FindDocs node
    if len(finddocs_nodes) == 0:
        logger.info("No FindDocs node found... *_actual columns will not be filled")
        last_node = None
    if len(finddocs_nodes) == 1:
        last_node = finddocs_nodes[0]
    else:
        logger.info(f"Found {len(finddocs_nodes)} FindDocs node(s)... picking the last one to fill *_actual columns")
        max_datetime = max(n.start_time for n in finddocs_nodes)
        last_node = next(n for n in finddocs_nodes if n.start_time == max_datetime)
    return last_node


def run_use_case(model: Runnable, eval_use_case: NamedTuple, output_df: pd.DataFrame) -> None:
        logger.info(f"Running model on use case {eval_use_case}")

        # run model on use case query
        with tracing_v2_enabled() as langchain_tracer:
            model.invoke({
                "chat_history": [],
                "question": eval_use_case.query,
            })

        # retrieve last FindDocs node
        last_node = find_last_finddocs_node(langchain_tracer, output_df, eval_use_case)

        if last_node:
            # extract inputs from last FindDocs node
            inputs_dict = (
                last_node.inputs if args.use_model_llm
                else json.loads(last_node.inputs["input"].replace("'", '"'))
            )
            inputs_dict = {
                k: None if isinstance(v, str) and v.lower() == 'none' else v
                for k, v in inputs_dict.items()
            }
            # extract outputs from last FindDocs node
            outputs_list = last_node.outputs["output"]
            if not args.use_model_llm:
                # make string valid for json to load as list of strings/documents
                outputs_list = outputs_list.replace('"', "'")
                outputs_list = outputs_list.replace("'<doc ", '"<doc ')
                outputs_list = outputs_list.replace("</doc>'", '</doc>"')
                outputs_list = outputs_list.replace("\\'", "'")
                outputs_list = json.loads(outputs_list)
            # fill *_actual columns in output_df with last FindDocs node
            if "state" in inputs_dict:
                output_df.loc[output_df.index == eval_use_case.Index, "state_filter_actual"] = inputs_dict["state"]
            if "county" in inputs_dict:
                output_df.loc[output_df.index == eval_use_case.Index, "county_filter_actual"] = inputs_dict["county"]
            if "commodity" in inputs_dict:
                output_df.loc[output_df.index == eval_use_case.Index, "commodity_filter_actual"] = inputs_dict["commodity"]
            if "doc_category" in inputs_dict:
                output_df.loc[output_df.index == eval_use_case.Index, "doc_category_filter_actual"] = inputs_dict["doc_category"]
            if len(outputs_list) >= 1:
                output_df.loc[output_df.index == eval_use_case.Index, "retrieved_doc1_actual"] = extract_s3_key(outputs_list[0])
            if len(outputs_list) >= 2:
                output_df.loc[output_df.index == eval_use_case.Index, "retrieved_doc2_actual"] = extract_s3_key(outputs_list[1])
            if len(outputs_list) >= 3:
                output_df.loc[output_df.index == eval_use_case.Index, "retrieved_doc3_actual"] = extract_s3_key(outputs_list[2])


def evaluate_use_case(output_df: pd.DataFrame) -> None:
    # evaluate use case, filter-wise
    def _get_filter_match(col_filter_actual: str, col_filter_expected: str) -> pd.Series:
        # case insensitive check between each actual and expected filter
        mask_both_na = (
            output_df[col_filter_actual].isna()
            & output_df[col_filter_expected].isna()
        )
        mask_both_equal = (
            output_df[col_filter_actual].str.lower() == output_df[col_filter_expected].str.lower()
        )
        return mask_both_na | mask_both_equal
    output_df["state_filter_match"] = _get_filter_match("state_filter_actual", "state_filter_expected")
    output_df["county_filter_match"] = _get_filter_match("county_filter_actual", "county_filter_expected")
    output_df["commodity_filter_match"] = _get_filter_match("commodity_filter_actual", "commodity_filter_expected")
    output_df["doc_category_filter_match"] = _get_filter_match("doc_category_filter_actual", "doc_category_filter_expected")

    # evaluate use case, retrieved documents-wise
    def _get_retrieved_doc_match(retrieved_doc_expected_col: str) -> pd.Series:
        # check that provided expected retrieved document is either
        # - NA
        # or
        # - equal to any of the actual retrieved documents (case sensitive)
        return (
            (output_df[retrieved_doc_expected_col].isna())
            |
            (output_df[retrieved_doc_expected_col] == output_df["retrieved_doc1_actual"])
            |
            (output_df[retrieved_doc_expected_col] == output_df["retrieved_doc2_actual"])
            |
            (output_df[retrieved_doc_expected_col] == output_df["retrieved_doc3_actual"])
        )
    output_df["retrieved_doc1_match"] = _get_retrieved_doc_match("retrieved_doc1_expected")
    output_df["retrieved_doc2_match"] = _get_retrieved_doc_match("retrieved_doc2_expected")
    output_df["retrieved_doc3_match"] = _get_retrieved_doc_match("retrieved_doc3_expected")

    # compute evaluation score
    match_cols = [
        "state_filter_match",
        "county_filter_match",
        "commodity_filter_match",
        "doc_category_filter_match",
        "retrieved_doc1_match",
        "retrieved_doc2_match",
        "retrieved_doc3_match",
    ]
    output_df["eval_score"] = output_df[match_cols].mean(axis=1)


def extract_s3_key(doc: str) -> str:
    s3_key = doc.split("s3_key='")[1].split("' url='")[0]
    return s3_key


def get_output_df(eval_df: pd.DataFrame) -> pd.DataFrame:
    output_df = eval_df.copy()
    cols_to_add = [
        col.replace("_expected", "_actual")
        for col in eval_df.columns
        if col.endswith("_expected")
    ]
    cols_to_add.append("nb_of_FindDocs_nodes_actual")
    output_df[cols_to_add] = None
    return output_df


def get_output_path(eval_path: str, use_model_llm: bool) -> str:
    output_root, output_ext = os.path.splitext(eval_path)
    output_root += "__model_llm" if use_model_llm else "__model_openai_functions"
    output_root += f"__{datetime.now().isoformat()}"
    return output_root + output_ext


if __name__ == "__main__":
    # parse args
    args = parse_args()
    logger.info(f"Evaluating croptalk's document retrieval capacity, using config: {args}\n")

    # read eval CSV into a df
    eval_df = pd.read_csv(args.eval_path, header=0, dtype=str)
    logger.info(f"Number of use cases to evaluate: {len(eval_df)}")

    # create output df
    output_df = get_output_df(eval_df)
    logger.info("Creating output_df")

    # load model
    logger.info("Loading model")
    if args.use_model_llm:
        from croptalk.model_llm import model
        memory = None
    else:
        from croptalk.model_openai_functions import model, memory

    # run model on each use case
    for eval_use_case in eval_df.itertuples(name="EvalUseCase"):
        if memory:
            memory.clear()
        run_use_case(model, eval_use_case, output_df)

    # evaluate each use case
    evaluate_use_case(output_df)

    # save output_df
    output_path = get_output_path(args.eval_path, args.use_model_llm)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Evaluation report/dataframe saved here: {output_path}")
