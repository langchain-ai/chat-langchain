import difflib
import logging
import os
import re
from typing import List, Dict
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, text

from croptalk.utils import read_pdf_from_s3, remove_long_words

load_dotenv("secrets/.env.secret")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

postgres_uri = os.environ.get('POSTGRES_URI_READ_ONLY')
DB = SQLDatabase.from_uri(postgres_uri)

load_dotenv("secrets/.env.secret")
load_dotenv("secrets/.env.shared")

COMMODITY_LIST = ['Wheat', 'Pecans', 'Cotton', 'Peaches', 'Corn', 'Peanuts', 'Whole Farm Revenue Protection',
                  'Soybeans', 'Pasture,Rangeland,Forage', 'Sesame', 'Controlled Environment', 'Apiculture', 'Hemp',
                  'Micro Farm', 'Blueberries', 'Oats', 'Fresh Market Sweet Corn', 'Grain Sorghum', 'Potatoes',
                  'Oysters', 'Triticale', 'Cucumbers', 'Canola', 'Popcorn', 'Fresh Market Tomatoes', 'Feeder Cattle',
                  'Fed Cattle', 'Cattle', 'Weaned Calves', 'Swine', 'Milk', 'Dairy Cattle', 'Forage Production',
                  'Dry Peas', 'Barley', 'Cabbage', 'Onions', 'Cotton Ex Long Staple', 'Chile Peppers', 'Dry Beans',
                  'Apples', 'Pistachios', 'Grapefruit', 'Lemons', 'Tangelos', 'Oranges', 'Mandarins/Tangerines', 'Rice',
                  'Hybrid Seed Rice', 'Grapes', 'Forage Seeding', 'Walnuts', 'Almonds', 'Prunes', 'Safflower',
                  'Cherries', 'Processing Cling Peaches', 'Kiwifruit', 'Olives', 'Tomatoes', 'Fresh Apricots',
                  'Processing Apricots', 'Pears', 'Raisins', 'Table Grapes', 'Figs', 'Plums', 'Alfalfa Seed',
                  'Strawberries', 'Tangelo Trees', 'Orange Trees', 'Grapefruit Trees', 'Lemon Trees',
                  'Fresh Nectarines', 'Processing Freestone', 'Fresh Freestone Peaches', 'Mandarin/Tangerine Trees',
                  'Pomegranates', 'Sugar Beets', 'Grapevine', 'Cultivated Wild Rice', 'Mint', 'Avocados', 'Caneberries',
                  'Millet', 'Sunflowers', 'Annual Forage', 'Nursery (NVS)', 'Silage Sorghum', 'Hybrid Sweet Corn Seed',
                  'Cigar Binder Tobacco', 'Cigar Wrapper Tobacco', 'Sweet Corn', 'Processing Beans', 'Green Peas',
                  'Flue Cured Tobacco', 'Tangors', 'Peppers', 'Sugarcane', 'Macadamia Nuts', 'Macadamia Trees',
                  'Banana', 'Coffee', 'Papaya', 'Banana Tree', 'Coffee Tree', 'Papaya Tree', 'Hybrid Popcorn Seed',
                  'Mustard', 'Grass Seed', 'Flax', 'Hybrid Corn Seed', 'Pumpkins', 'Burley Tobacco',
                  'Hybrid Sorghum Seed', 'Camelina', 'Dark Air Tobacco', 'Fire Cured Tobacco', 'Sweet Potatoes',
                  'Maryland Tobacco', 'Cranberries', 'Clams', 'Buckwheat', 'Rye', 'Fresh Market Beans', 'Clary Sage',
                  'Hybrid Vegetable Seed', 'Cigar Filler Tobacco', 'Tangerine Trees', 'Lime Trees']


@tool("get-SP-doc")
def get_sp_document(state: Optional[str] = None,
                    county: Optional[str] = None,
                    commodity: Optional[str] = None,
                    year: Optional[int] = None):
    """
    County specific insurance question should be answered with this tool. It allows to retrieve
    information the specific way a policy works within a county (prices, dates, rules).

    Example questions this tool can answer:
    What are the grade discount wheat classes in Alabama, Autauga?
    What is the final planting date for Virginia type peanuts in Baldwin County, Alabama, for the 2023 crop year?
    What is the practice code for organic (certified) irrigated cotton in Cleburne County, Alabama, for the 2024 crop year?

    Args:
    state : name of state
    county_name: name of county provided
    commodity_name: name of commodity
    year: year of document

    Returns: str
    """

    if all([state, county, commodity]):
        # use chain to retrieve correct commodity info
        # we use a chain in this case because commodity is not easily retrieved by the agent in the correct format
        commodity = difflib.get_close_matches(commodity, COMMODITY_LIST, cutoff=0.1)[0]
        county = county.lower().replace("county", "")

        # Define your database connection URL
        db_url = os.environ["POSTGRES_URI"]

        # Create the SQLAlchemy engine
        engine = create_engine(db_url)

        # Define your SQL query
        sql_query = f"""
           SELECT s3_key, year
           FROM sp_files
           WHERE commodity_name = '{commodity.lower()}' AND
           state_name = '{state.lower()}' AND 
           county_name = '{county.lower()}'
           """

        # Execute the SQL query
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()

        s3_keys = pd.DataFrame(rows)
        if s3_keys.empty:
            return (f"My search results indicate that there is no Special Provision (SP) document for the "
                    f"corresponding commodity ({commodity}), state ({state}),and county ({county}). \b"
                    "Make sure you are providing available commodity, state and county.")

        if not year:
            # take document in latest year
            s3_keys = s3_keys[s3_keys["year"] == s3_keys["year"].max()]
        else:
            years = s3_keys["year"].unique()
            s3_keys = s3_keys[s3_keys["year"] == year]
            if s3_keys.empty:
                return ("My search results indicate that there are no Special Provision (SP) documents for the county,"
                        " state and commodity you requested. However, there are none for this specific year. "
                        f"Here is the list of available years for the requested SP document : {years}")

        doc_link = s3_keys["s3_key"].values[0]

        message = f"The Special provision (SP) document for "
        if year:
            message += f"year {year},"

        sp_content = read_pdf_from_s3("croptalk-spoi", doc_link)
        sp_content = remove_long_words(sp_content)

        message += (f"{commodity}, {state} and {county} county can be found at the following "
                    f"link : https://croptalk-spoi.s3.us-east-2.amazonaws.com/{doc_link}")
        message += "Here is the document content : " + sp_content

        return message

    else:

        # format message to hint user to add appropriate information
        var_names = {"state": state, "commodity_name": commodity, "county": county}
        missing_var_msg = ("Please specify the following to obtain the specific Special Provision (SP) document "
                           "you are requesting : ")
        missing_var_list = [i for i, j in var_names.items() if j is None]

        # depending on list length, format message differently
        if len(missing_var_list) == 1:
            missing_var_msg += missing_var_list[0]
        else:
            missing_var_msg += ", ".join(missing_var_list)

        return missing_var_msg


class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = []

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can
         record it as the final sql"""

        if action.tool in ["sql_db_query_checker", "sql_db_query"]:
            self.sql_result.append(action.tool_input)


@tool("get-SOB-metrics-using-SQL-agent")
def get_sob_metrics_sql_agent(input: str) -> str:
    """
    This tool is used to query insurance statistics. This tool can find data on policy sold, indemnifications,
    liability, cost to grower and premiums in function of the county, insurance plan and year.

    Here are some example of questions this tool can answer (amongst other):
    - What is the total of policies sold in the state of New York for the WFRP policy in year 2023
    - Which insurance plan has the highest average expected payout in 2023
    - What is the average loss ratio by insurance plan name and year
    - What is the average cost to grower under the APH policy for walnuts in Fresno county in California
    - What is the distribution of policy sold amongst counties for the RP policy for the state of Kansas

    Args:
        input: user query

    Returns: agent answer with SQL data

    Returns: str
    """
    try:
        # this does not work with gpt-4
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        agent = create_sql_agent(
            llm=llm,
            db=DB,
            prompt=full_prompt,
            tools=[validate_query],
            verbose=True,
            agent_type="openai-tools",
        ).with_config(run_name="SQLAgentExecutor")

        handler = SQLHandler()

        response = agent.invoke({"input": input,
                                 "top_k": 3,
                                 "dialect": "SQLite",
                                 "agent_scratchpad": [],
                                 }, {"callbacks": [handler]})["output"]

        return f"SQL query : {handler.sql_result}, Output :  {response}"

    except SQLStatementNotAllowed:
        return "Only read operations are allowed."

    except Exception as e:
        return "There was an Error in SQL tool."


class SQLStatementNotAllowed(Exception):
    """Raise for my specific kind of exception"""


@tool("validate-sql_query")
def validate_query(output: str) -> None:
    """
    This tool should be used for EVERY sql query. It is meant to validate that the query does not contain
    any dangerous statements.

    Args:
        output: SQL query

    Returns: None
    """
    # Define the regex pattern to match SQL operations
    pattern = r'\b(?:INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b'

    # Use re.search to check if the pattern is found in the query
    match = re.search(pattern, output, re.IGNORECASE)  # Use IGNORECASE flag to ignore case sensitivity

    if match:
        raise SQLStatementNotAllowed


@tool("wfrp-commodities-tool")
def get_wfrp_commodities(reinsurance_yr: str, state_code: str, county_code: str) -> Optional[str]:
    """
    This tool's sole purpose is to find commodities associated with the Whole Farm Revenue Protection (WFRP) insurance
    program for a given reinsurance year, state code and county code. This tool answers questions such as : What are
    the available commodities for a specific county and state?

    It is NOT related to SP documents.

    To use this tool, you must provide all three arguments [reinsurance_yr, state_code and county_code]

    Args:
        reinsurance_yr (str): year of reinsurance, ex : "2024"
        state_code: (str) : state code, ex : "06" for California
        county_code: (str) : county code, ex : "001"

    Returns: Optional[str]
    """
    url = f"http://dev.lookup-api.cropguard.online/v1/lookups/GetWFRPCommodities"
    headers = {
        'accept': 'application/json',
        'api-key': os.environ["CROPGUARD_API_KEY"],
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json={"ReinsuranceYear": reinsurance_yr,
                                                         "StateCode": state_code,
                                                         "CountyCode": county_code})

    if response.status_code == 200:
        print("API call successful")
        response_data = response.json()
        # Process the response data as needed
        commodity_list = [i["AGR Commodity Name"] for i in response_data]
        return f"Available commodities for the WFRP program (year {reinsurance_yr}, state_code {state_code}, and " \
               f"county_code {county_code}) are : " + str(commodity_list)

    else:
        print("API call failed with status code:", response.status_code)
        return None


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
            "query": "SELECT SUM(policies_indemnified_count) * 100.0 / SUM(policies_sold_count) FROM sob_all_years WHERE county_name = 'Washakie' AND state_abbreviation = 'WY' AND insurance_pla_name_abbreviation == 'APH'"

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

TOOLS = [get_sob_metrics_sql_agent, get_wfrp_commodities, get_sp_document]
RENDERED_TOOLS = render_text_description(TOOLS)

TOOL_PROMPT = f"""You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{RENDERED_TOOLS} """ + """
Given the user questions, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

Do not use tools if they are not necessary.

This is the question you are being asked : {question}

"""

ROUTE_TEMPLATE = f""" \
Act as a classifier with the task of distinguishing between two topics from a question: `tools` or `other`.
`tools` is defined as any question related to Special Provision (SP) documents, the list of available commodities within
Whole Farm Revenue Protection (WFRP) and questions related to insurance market data (Summary of business or SOB).
Here is the list of rendered tools {RENDERED_TOOLS}

Example questions of the `tools` topic include (but is not limited to) the following examples :

""" + """
- what are the available commodities for WFRP for Butte county in California
- find me the special provision document related to Oranges in Yakima, Washington for the year 2024
- SP document for corn in Butte, California, 2022
- get me the SP document for corn in Iowa for Pottawattamie county

Do not respond with more than word.

<question>
{question}
</question>

Classification:"""
