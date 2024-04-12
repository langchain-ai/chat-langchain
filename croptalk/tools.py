import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sqlalchemy import create_engine, text

from croptalk.utils import initialize_llm
from croptalk.prompt_tools import full_prompt
from croptalk.prompts_llm import COMMODITY_TEMPLATE_TOOL

postgres_uri = os.environ.get('POSTGRES_URI_READ_ONLY')
DB = SQLDatabase.from_uri(postgres_uri)

load_dotenv("secrets/.env.secret")
load_dotenv("secrets/.env.shared")


@tool("get-SP-doc")
def get_sp_document(state: Optional[str] = None,
                    county: Optional[str] = None,
                    commodity: Optional[str] = None,
                    year: Optional[int] = None):
    """
    This tool is used to query the SP document from a database along state, county, commodity and year.
    For instance, a user might ask :
    - find me the SP document related to Oranges in Yakima, Washington for the year 2024
    - SP document for corn in Butte, California, 2022

    Args:
    state : name of state
    county_name: name of county provided
    commodity_name: name of commodity
    year: year of document

    """
    if all([state, county, commodity, year]):
        # use chain to retrieve correct commodity info
        # we use a chain in this case because commodity is not easily retrieved by the agent in the correct format
        llm = initialize_llm(os.environ["MODEL_NAME"])
        commodity_prompt = PromptTemplate.from_template(COMMODITY_TEMPLATE_TOOL)
        commodity_chain = commodity_prompt | llm | StrOutputParser()
        commodity = commodity_chain.invoke({'question': commodity})

        # Define your database connection URL
        db_url = os.environ["POSTGRES_URI"]

        # Create the SQLAlchemy engine
        engine = create_engine(db_url)

        # Define your SQL query
        sql_query = f"""
        SELECT s3_key 
        FROM sp_files
        WHERE year = {year} AND 
        commodity_name = '{commodity.lower()}' AND
        state_name = '{state.lower()}' AND 
        county_name = '{county.lower()}'
        """

        # Execute the SQL query
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()

        if not rows:
            return (f"My search results indicate that there is no SP document for the corresponding year ({year}), "
                    f"commodity ({commodity}), state ({state}),and county ({county}). \b"
                    "Make sure you are providing available year, commodity, state and county.")
        return (f"Here is the link to the SP document you're looking for : "
                f"https://croptalk-spoi.s3.us-east-2.amazonaws.com/SPOI/{rows[0][0]}")

    else:
        # format message to hint user to add appropriate information
        var_names = {"state": state, "commodity_name": commodity, "year": year, "county": county}
        missing_var_msg = "Please specify the following to obtain the specific SP document you are requesting : "
        missing_var_list = [i for i, j in var_names.items() if j is None]

        # depending on list length, format message differently
        if len(missing_var_list) == 1:
            missing_var_msg += missing_var_list[0]
        else:
            missing_var_msg += ", ".join(missing_var_list)

        return missing_var_msg


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
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        agent = create_sql_agent(
            llm=llm,
            db=DB,
            prompt=full_prompt,
            tools=[validate_query],
            verbose=True,
            agent_type="openai-tools",
        )

        return agent.invoke({"input": input,
                             "top_k": 3,
                             "dialect": "SQLite",
                             "agent_scratchpad": [],
                             })["output"]

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

    Returns:

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
    This tool is used to find commodities associated with the Whole Farm Revenue Protection (WFRP) insurance program
    for a given reinsurance year, state code and county code. Do not use this tool unless you are explicitly asked
    about available commodities for the WFRP program.

    To use this tool, you must provide all three arguments [reinsurance_yr, state_code and county_code]
    Do not make up those arguments if they are not provided explicitly.

    Args:
        reinsurance_yr (str): year of reinsurance
        state_code: (str) : state code
        county_code: (str) : county code

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


tools = [get_sob_metrics_sql_agent, get_wfrp_commodities, get_sp_document]
