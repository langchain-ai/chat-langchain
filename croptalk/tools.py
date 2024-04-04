from typing import Optional, Union

import requests
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
# from croptalk.load_data import SOB
import os
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv("secrets/.env.secret")
load_dotenv("secrets/.env.shared")


def initialize_llm(model):
    return ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0,
    )


def create_commodity_tool_chain():
    llm = initialize_llm(os.environ["MODEL_NAME"])

    commodity_template_tool = """\
    Given the following words identify whether is it matches to any of the following commodities. 
    If it is, extract the relevant commodity and return it. If it is not, return 'None'.

    Commodities: \
    ['Wheat', 'Pecans', 'Cotton', 'Peaches', 'Corn', 'Peanuts', 'Whole Farm Revenue Protection', 'Soybeans', 'Pasture,Rangeland,Forage', 'Sesame', 'Controlled Environment', 'Apiculture', 'Hemp', 'Micro Farm', 'Blueberries', 'Oats', 'Fresh Market Sweet Corn', 'Grain Sorghum', 'Potatoes', 'Oysters', 'Triticale', 'Cucumbers', 'Canola', 'Popcorn', 'Fresh Market Tomatoes', 'Feeder Cattle', 'Fed Cattle', 'Cattle', 'Weaned Calves', 'Swine', 'Milk', 'Dairy Cattle', 'Forage Production', 'Dry Peas', 'Barley', 'Cabbage', 'Onions', 'Cotton Ex Long Staple', 'Chile Peppers', 'Dry Beans', 'Apples', 'Pistachios', 'Grapefruit', 'Lemons', 'Tangelos', 'Oranges', 'Mandarins/Tangerines', 'Rice', 'Hybrid Seed Rice', 'Grapes', 'Forage Seeding', 'Walnuts', 'Almonds', 'Prunes', 'Safflower', 'Cherries', 'Processing Cling Peaches', 'Kiwifruit', 'Olives', 'Tomatoes', 'Fresh Apricots', 'Processing Apricots', 'Pears', 'Raisins', 'Table Grapes', 'Figs', 'Plums', 'Alfalfa Seed', 'Strawberries', 'Tangelo Trees', 'Orange Trees', 'Grapefruit Trees', 'Lemon Trees', 'Fresh Nectarines', 'Processing Freestone', 'Fresh Freestone Peaches', 'Mandarin/Tangerine Trees', 'Pomegranates', 'Sugar Beets', 'Grapevine', 'Cultivated Wild Rice', 'Mint', 'Avocados', 'Caneberries', 'Millet', 'Sunflowers', 'Annual Forage', 'Nursery (NVS)', 'Silage Sorghum', 'Hybrid Sweet Corn Seed', 'Cigar Binder Tobacco', 'Cigar Wrapper Tobacco', 'Sweet Corn', 'Processing Beans', 'Green Peas', 'Flue Cured Tobacco', 'Tangors', 'Peppers', 'Sugarcane', 'Macadamia Nuts', 'Macadamia Trees', 'Banana', 'Coffee', 'Papaya', 'Banana Tree', 'Coffee Tree', 'Papaya Tree', 'Hybrid Popcorn Seed', 'Mustard', 'Grass Seed', 'Flax', 'Hybrid Corn Seed', 'Pumpkins', 'Burley Tobacco', 'Hybrid Sorghum Seed', 'Camelina', 'Dark Air Tobacco', 'Fire Cured Tobacco', 'Sweet Potatoes', 'Maryland Tobacco', 'Cranberries', 'Clams', 'Buckwheat', 'Rye', 'Fresh Market Beans', 'Clary Sage', 'Hybrid Vegetable Seed', 'Cigar Filler Tobacco', 'Tangerine Trees', 'Lime Trees']
    Words: {question}
    commodity: """

    commodity_prompt = PromptTemplate.from_template(commodity_template_tool)
    return commodity_prompt | llm | StrOutputParser()


@tool("get_SP_doc")
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
    insurance_plan_name: provided plan name abbreviation

    Returns: str
    """

    if all([state, county, commodity, year]):
        # use chain to retrieve correct commodity info
        commodity_chain = create_commodity_tool_chain()
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
            return ("My search results indicate that there is no SP document for the corresponding year, commodity, "
                    "state and county. \b"
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


@tool("get_SOB_metrics")
def get_sob_metrics_for_crop_county(state_abbreviation: str, county_name: str, commodity_name: str,
                                    insurance_plan_name: str, metric: str) -> str:
    """
    This tool is used to query the summary of business data (SOB) to retrieve insurance program metrics by coverage level.

    Those metric can be percentage liability indemnified (pct_liability_indemnified), cost to grower (cost_to_grower),
    the number of policies sold (policies_sold_count) or the premium per quantity (premium_per_quantity)

    For instance, a user might ask : "What is the percentage of policies indemnified for Pierce county in North Dakota,
    for sunflowers under the APH program?". The tool will answer this question with the following arguments:

    Args:
        state_abbreviation: two letter string of State abbreviation (California -> CA, North Dakota -> ND and so on)
        county_name: name of county provided
        commodity_name: name of commodity
        insurance_plan_name: provided plan name abbreviation
        metric : choice from ["pct_liability_indemnified", "cost_to_grower", "policies_sold_count", "premium_per_quantity"]

    Returns: the tool returns a string which gives the coverage level with their
            associated percentage indemnified statistic

    """

    SOB = pd.DataFrame()
    sob = SOB[SOB["commodity_year"] == 2023]

    sob = sob[
        (sob["state_abbreviation"] == state_abbreviation) &
        (sob["county_name"] == county_name) &
        (sob["commodity_name"] == commodity_name.lower()) &
        (sob["insurance_plan_name_abbreviation"] == insurance_plan_name)
        ]

    return "In 2023, we observe the following data" + str(
        [{"coverage_level": i, metric: j} for i, j in zip(list(sob["coverage_level"]), list(sob[metric]))]
    )


@tool("wfrp-commodities-tool")
def get_wfrp_commodities(reinsurance_yr: str, state_code: str, county_code: str) -> Optional[str]:
    """
    This tool is used to find commodities associated with the Whole Farm Revenue Protection (WFRP) insurance program
     for a given reinsurance year, state code and county code.

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
        return str([i["AGR Commodity Name"] for i in response_data])
    else:
        print("API call failed with status code:", response.status_code)
        return None


tools = [get_wfrp_commodities, get_sp_document]
