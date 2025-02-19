.PHONY: start
start:
	poetry run uvicorn --app-dir=backend main:app --reload --port 8070

.PHONY: format
format:
	poetry run ruff format .
	poetry run ruff --select I --fix .

.PHONY: lint
lint:
	poetry run ruff .
	poetry run ruff format . --diff
	poetry run ruff --select I .

