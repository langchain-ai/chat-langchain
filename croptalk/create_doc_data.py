import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv("secrets/.env.secret")

# Define your database connection URL
db_url = os.environ["POSTGRES_URI"]

# Create the SQLAlchemy engine
engine = create_engine(db_url)


def query_sql_data(query):
    # Execute the SQL query
    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = result.fetchall()

    return pd.DataFrame(data=rows)


if __name__ == '__main__':
    # TODO this should be a SQL statement using the data from the generated weaviate_collection

    # get chunks data, get unique s3 keys
    chunks = pd.read_csv("data/total_chunks_in_weaviate_collection.csv")
    sp_files = chunks[(chunks["doc_category"] == "SP") &
                      (chunks["year"] == 2024)].drop_duplicates("s3_key")

    # renaming and preprocessing to make sure of merge compatibility
    sp_files = sp_files.rename(columns={"state": "state_code", "county": "county_code", "commodity": "commodity_code"})
    sp_files["state_code"] = sp_files["state_code"].apply(lambda x: str(x).zfill(2))
    sp_files["county_code"] = sp_files["county_code"].apply(lambda x: str(x).zfill(3))
    sp_files["commodity_code"] = sp_files["commodity_code"].apply(lambda x: str(x).zfill(4))

    # get state data
    state = query_sql_data(query="""
        SELECT "State Code" as state_code, "State Name" as state_name
        FROM state
        """)
    state["state_name"] = state["state_name"].apply(lambda x: x.lower())

    # get county data
    county = query_sql_data(query="""
        SELECT "State Code" as state_code, "County Code" as county_code, "County Name" as county_name
        FROM county
        """)
    county["county_name"] = county["county_name"].apply(lambda x: x.lower())

    # get commodity data
    commodity = query_sql_data(query="""
        SELECT "Commodity Code" as commodity_code, "Commodity Name" as commodity_name
        FROM commodity
        """)
    commodity["commodity_name"] = commodity["commodity_name"].apply(lambda x: x.lower())

    # merge sp_files with names for state, county and commodity
    sp_files = pd.merge(sp_files, state, on="state_code")
    sp_files = pd.merge(sp_files, county, on=["state_code", "county_code"])
    sp_files = pd.merge(sp_files, commodity, on="commodity_code")

    # only load relevant columns into postgres
    table_name = 'sp_files'
    cols_to_include = ['title', 'content', 'commodity_name', 'state_name', 'county_name', 'doc_category', 'year', 's3_key']

    sp_files[cols_to_include].to_sql(table_name,
                                     engine,
                                     if_exists='replace',
                                     index=False)

