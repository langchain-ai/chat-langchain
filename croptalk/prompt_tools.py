import os
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv("secrets/.env.secret")


def get_sob_sql_query_examples() -> List[Dict]:
    return [
        {
            "input": "What is the total of policies sold in the state of New York for the WFRP policy in year 2023",
            "query": "SELECT SUM(policies_sold_count) AS total_policies_sold FROM sob_all_years WHERE state_abbreviation = 'NY' AND insurance_plan_name_abbreviation = 'WFRP' AND commodity_year = 2023",
        },
        {
            "input": "What is the total number of policies sold for Bee county in Texas, for corn, for the RP program, for 0.7 coverage level for year 2023",
            "query": "SELECT SUM(policies_sold_count) FROM sob_all_years WHERE county_name = 'Bee' AND state_abbreviation = 'TX' AND commodity_name = 'corn' AND coverage_level = 0.7 AND commodity_year = 2023",
        },
        {
            "input": "What is the average cost to grower under the APH policy for walnuts in Fresno county in California",
            "query": "SELECT SUM(cost_to_grower * policy_sold_count)/SUM(cost_to_grower) A AS average_cost_to_grower FROM sob_all_years WHERE county_name = 'Fresno' AND state_abbreviation = 'CA' AND insurance_plan_name_abbreviation = 'APH' AND commodity_name = 'walnuts'",
        },
        {
            "input": "What is the average loss ratio by insurance plan name and year ",
            "query": "SELECT SUM(loss_ratio * policy_sold_count)/SUM(loss_ratio) AS average_loss_ratio FROM sob_all_years GROUP BY commodity_year, insurance_pla_name_abbreviation"
        },

        {
            "input": "Which insurance plan has the highest average expected payout in 2023",
            "query": "SELECT SUM(expected_payout * policy_sold_count)/SUM(expected_payout) AS average_expected_payout FROM sob_all_years WHERE commodity_year == 2023 ORDER BY average_expected_payout DESC LIMIT 1"

        },
        {
            "input": "What is the cost to grower under the APH policy for walnuts in Fresno county in California in 2023",
            "query": "SELECT SUM(cost_to_grower * policy_sold_count)/SUM(cost_to_grower) AS average_cost_to_grower FROM sob_all_years WHERE commodity_year == 2023 AND county_name == 'Fresno' and state_abbreviation = 'CA' ORDER BY average_cost_to_grower DESC LIMIT 1"

        },
        {
            "input": "What is the percentage of policies indemnified for Washakie county in Wyoming "
                    "for Corn under the APH program",
            "query":  "SELECT SUM(policies_indemnified_count) * 100.0 / SUM(policies_sold_count) FROM sob_all_years WHERE county_name = 'Washakie' AND state_abbreviation = 'WY' AND insurance_pla_name_abbreviation == 'APH'"

        },

    ]


system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

Always use the sob_all_years table. 

If a question is about an average, make sure to weight the metric by policy_sold_count.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

examples = get_sob_sql_query_examples()

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
    FAISS,
    k=5,
    input_keys=["input"],
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


# layer to check regex for DML statements
# todo ask : llm to provide potential questions based on the data or queries
# provide sql statements and generate the question
# examples of queries which are not possible to answer
# threshold for similarity with the example queries
# prompt : explain how the result are retrieved (which sort of operation was done)
# idea : use regex to validate if query is similar
