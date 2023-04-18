.PHONY: start
start:
	python3 main.py

.PHONY: format
format:
	black .
	isort .