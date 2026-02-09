.PHONY: start, format, lint

format:
	uv run ruff format .
	uv run ruff check --select I --fix .

lint:
	uv run ruff format . --diff
	uv run ruff check --select I .

