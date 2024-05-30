.PHONY: start
start:
	langgraph up --watch

.PHONY: format
format:
	poetry run ruff format .
	poetry run ruff --select I --fix .

.PHONY: lint
lint:
	poetry run ruff .
	poetry run ruff format . --diff
	poetry run ruff --select I .

