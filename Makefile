.PHONY: start, format, lint

format:
	poetry run ruff format .
	poetry run ruff --select I --fix .

lint:
	poetry run ruff .
	poetry run ruff format . --diff
	poetry run ruff --select I .

