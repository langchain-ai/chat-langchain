.PHONY: start start-chat-gpt
start:
	uvicorn main:app --reload --port 9000
start-chat-gpt:
	uvicorn main:app --reload --port 9000

.PHONY: format
format:
	black .
	isort .