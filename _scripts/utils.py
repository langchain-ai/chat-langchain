import argparse
import os
from datetime import datetime
from typing import List

from langchain_core.tracers import Run


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-model-llm",
        help="Option which, when specified, tells the evaluation to use model_llm (i.e. use model_openai_functions when this option is not specified)",
        action='store_true',
    )

    parser.add_argument(
        "--eval_path",
        help="CSV file path that contains evaluation use cases",
        default=None,
        action='store_true',
    )
    return parser.parse_args()


def get_output_path(eval_path: str, use_model_llm: bool) -> str:
    now_string = datetime.now().isoformat().replace(":", "")
    output_root, output_ext = os.path.splitext(eval_path)
    output_root += "__model_llm" if use_model_llm else "__model_openai_functions"
    output_root += f"__{now_string}"
    return output_root + output_ext
