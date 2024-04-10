import argparse
from datetime import datetime
import logging
import os
from typing import List, NamedTuple

import json
from langchain_core.tracers.context import tracing_v2_enabled
from croptalk.tools import tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":

    # load model
    logger.info("Loading model")
    from croptalk.model_llm import model
    #from croptalk.model_openai_functions import model

    with tracing_v2_enabled() as langchain_tracer:
        # model.invoke({
        #     "chat_history": [],
        #     "rendered_tools": rendered_tools,
        #     "question": "What are the livestock insured with WFRP for reinsurance year 2024,"
        #                 " state code 04 and county code 001 ",
        # })
        #
        #model.invoke({
        #   "chat_history": [],
        #    "question": "Find me the SP document for corn, in Washington"
        #})

        output = model.invoke({
            "chat_history": [],
            "question": "Find me the SP document for peanut, Monroe, Missouri in 2024"
        })
        print(output)

        output = model.invoke({
            "chat_history": [],
            "question": "Find me the SP document for almond, san joaquim, california in 2024"
        })
        print(output)

        # todo overwrite the other context if no available documents?

        #output = model.invoke({
        #    "chat_history": [],
        #    "question": "Find me the SP document for Soybeans, Missouri and Monroe county in 2024"
        #})
        #print(output)

        #model.invoke({
        #    "chat_history": [],
        #    "question": "Find me the SP document for Barley, Missouri and Monroe county in 1999"
        #})


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
