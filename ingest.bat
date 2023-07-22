@echo off

REM Attempt to download the site. If wget is not available this will fail.
wget -r -A.html https://api.python.langchain.com/en/latest/api_reference.html

REM Check error level of previous command and exit if non-zero.
if errorlevel 1 exit /b %errorlevel%

REM Run the Python script
python ingest.py
