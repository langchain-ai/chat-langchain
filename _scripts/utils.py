import argparse
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
        "--test-with_csv",
        help="Option which, when specified, uses the evaluation csv to evaluate the model",
        action='store_true',
    )

    parser.add_argument(
        "eval_path",
        help="CSV file path that contains evaluation use cases",
    )
    return parser.parse_args()
