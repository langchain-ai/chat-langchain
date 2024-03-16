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

if __name__ == "__main__":

    # load model
    logger.info("Loading model")
    from croptalk.model_llm import model

    rendered_tools = render_text_description(tools)
    with tracing_v2_enabled() as langchain_tracer:
        # model.invoke({
        #     "chat_history": [],
        #     "rendered_tools": rendered_tools,
        #     "question": "What are the livestock insured with WFRP for reinsurance year 2024,"
        #                 " state code 04 and county code 001 ",
        # })
        #
        model.invoke({
            "chat_history": [],
            "rendered_tools": rendered_tools,
            "question": "What is the percentage of policies indemnified for Washakie county in Wyoming "
                        "for sugar beets under the APH program"
        })

        # model.invoke({
        #     "chat_history": [],
        #     "rendered_tools": rendered_tools,
        #     "question": "What is the number of policies sold for Bee county in Texas, for corn, for the RP program"
        # })
        #
        #
        # model.invoke({
        #     "chat_history": [],
        #     "rendered_tools": rendered_tools,
        #     "question": "What is the number of policies sold for Bee county in Texas, for corn, "
        #                 "for the RP program, for 0.7 coverage level"
        # })

        # model.invoke({
        #     "chat_history": [],
        #     "rendered_tools": rendered_tools,
        #     "question": "What is the number of policies sold for Bee county in Texas"
        # })
