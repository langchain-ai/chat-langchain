# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
# Error if any command fails
set -e
wget -r -A https://aide.blank.app/hc/fr --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64)"
python3 ingest.py
