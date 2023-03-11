.PHONY: start
start:
	uvicorn main:app --reload --port 9000 --log-level debug

.PHONY: format
format:
	black .
	isort .