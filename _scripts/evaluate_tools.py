import argparse
from datetime import datetime
import logging
import os
from typing import List, NamedTuple

import json
from langchain_core.tracers.context import tracing_v2_enabled
from croptalk.tools import tools
from langchain.tools.render import render_text_description

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


if __name__ == "__main__":
    # parse args

    # load model
    logger.info("Loading model")
    from croptalk.model_llm import model

    rendered_tools = render_text_description(tools)
    with tracing_v2_enabled() as langchain_tracer:
        model.invoke({
            "chat_history": [],
            "rendered_tools": rendered_tools,
            "question": "What are the livestock insured with WFRP for reinsurance year 2024,"
                        " state code 04 and county code 001 ",
        })
