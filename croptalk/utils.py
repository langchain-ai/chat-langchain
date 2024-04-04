import pandas as pd
from sqlalchemy import text

from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

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


query = """
SELECT SUM(policies_indemnified_count) * 100.0 / SUM(policies_sold_count) FROM sob_all_years WHERE county_name = 'Washakie' AND state_abbreviation = 'WY'
"""
#ELECT SUM(indemnified_policies_count) * 100.0 / SUM(policies_sold_count) AS percentage_of_policies_indemnified FROM
df = query_sql_data(query)
