import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("secrets/.env.secret")

# Define your database connection URL
db_url = os.environ["POSTGRES_URI"]

# Create the SQLAlchemy engine
engine = create_engine(db_url)

# Define your SQL query
sql_query = """
SELECT * 
FROM sob_all_years
WHERE commodity_year >= 2010
"""

# Execute the SQL query
with engine.connect() as connection:
    result = connection.execute(text(sql_query))
    rows = result.fetchall()

SOB = pd.DataFrame(data=rows)
