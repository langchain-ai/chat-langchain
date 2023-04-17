#======================== builder ========================
FROM python:3.11.3-bullseye as builder

RUN python -m venv --copies /venv
RUN /venv/bin/pip install --upgrade pip

#======================== builder-venv ========================
FROM builder AS builder-venv
ENV POETRY_VERSION=1.3.2
RUN /venv/bin/pip install "poetry==$POETRY_VERSION"

COPY ./pyproject.toml ./poetry.lock ./

RUN /venv/bin/poetry config virtualenvs.create false \
    && /venv/bin/poetry config virtualenvs.in-project false \
    && /venv/bin/poetry install --no-root --only main    

#======================== multistage ========================
#FROM cgr.dev/chainguard/python
FROM python:3.11.3-slim-bullseye
COPY --from=builder-venv /venv /venv

COPY . /app
WORKDIR /app

# And then will start Gunicorn with Uvicorn
ENTRYPOINT ["/venv/bin/gunicorn"  , "-k", "uvicorn.workers.UvicornWorker", "-c", "app/docker-images/gunicorn_conf.py", "main:app"]
