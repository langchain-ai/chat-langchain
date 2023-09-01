.PHONY: start
start:
	uvicorn main:app --reload --port 8080

.PHONY: format
format:
	black .
	isort .