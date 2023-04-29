.PHONY: start
start:
	poetry run uvicorn main:app --reload --port 9000

.PHONY: format
format:
	poetry run black .
	poetry run isort .