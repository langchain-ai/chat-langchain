@echo off
rem Batch script to ingest data
rem This involves scraping the data from the web and then cleaning up and putting in Weaviate.
wget -r -A.html https://python.langchain.com/en/latest/
if %errorlevel% neq 0 exit /b %errorlevel%
python3 ingest.py
if %errorlevel% neq 0 exit /b %errorlevel%
