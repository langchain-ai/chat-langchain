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
