import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from croptalk.utils import query_sql_data

load_dotenv("secrets/.env.secret")

# Define your database connection URL
db_url = os.environ["POSTGRES_URI"]

# Create the SQLAlchemy engine
engine = create_engine(db_url)

SOB = query_sql_data(query="""
    SELECT * 
    FROM sob_all_years
    WHERE commodity_year >= 2010
    """)