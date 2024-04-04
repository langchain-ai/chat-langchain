import pandas as pd
from sqlalchemy import text

from croptalk.create_doc_data import engine


def query_sql_data(query):
    # Execute the SQL query
    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = result.fetchall()

    return pd.DataFrame(data=rows)
