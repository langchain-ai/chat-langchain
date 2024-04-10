import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from croptalk.create_doc_data import query_sql_data, engine, db_url

load_dotenv("secrets/.env.secret")

SOB = query_sql_data(query="""
    SELECT * 
    FROM sob_all_years
    WHERE commodity_year >= 2010
    """)
