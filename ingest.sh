# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
# Error if any command fails
set -e
wget -r -A.html https://api.python.langchain.com/en/latest
wget -r -A.html https://python.langchain.com/docs/get_started/introduction
python3 ingest.py