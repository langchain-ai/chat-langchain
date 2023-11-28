FROM python:3.11-buster

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

# Copy the pyproject.toml and poetry.lock files
COPY ./pyproject.toml ./poetry.lock* ./

# Install dependencies without the application's package
RUN poetry install --no-interaction --no-ansi --no-root --no-directory

# Copy the entire project directory
COPY . .

# Install the application's package
RUN poetry install  --no-interaction --no-ansi

CMD exec uvicorn main:app --host 0.0.0.0 --port 8080
