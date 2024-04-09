import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from croptalk.prompt_tools import full_prompt

load_dotenv("secrets/.env.secret")
postgres_uri = os.environ.get('POSTGRES_URI')
DB = SQLDatabase.from_uri(postgres_uri)


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

    """
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # agent = create_sql_agent(
        #     llm=llm,
        #     db=DB,
        #     prompt=full_prompt,
        #     verbose=True,
        #     agent_type="openai-tools",
        # )
        #
        # return agent.invoke({"input": input,
        #                      "top_k": 3,
        #                      "dialect": "SQLite",
        #                      "agent_scratchpad": [],
        #                      })["output"]

        def validate_query(output: str) -> str:
            # Define the regex pattern to match SQL operations
            pattern = r'\b(?:INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b'

            # Use re.search to check if the pattern is found in the query
            match = re.search(pattern, output, re.IGNORECASE)  # Use IGNORECASE flag to ignore case sensitivity

            if match:
                return "The query is using a forbidden operation"
            return output

        chain = create_sql_query_chain(llm, DB, prompt=full_prompt) | validate_query
        query = chain.invoke(
            {"input": input,
             "top_k": 3,
             "dialect": "SQLite",
             "agent_scratchpad": [],
             })

        return DB.run(query)

    except:
        return "There was an Error in SQL tool"


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


tools = [get_sob_metrics_sql_agent, get_wfrp_commodities]  #
