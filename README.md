# Helper

A thing - what helps people, search for stuff. It's a FastAPI project mainly, with Jinja2 templates. But we can change that...

# Python Setup

Install the correct version of python using [pyenv](https://github.com/pyenv/pyenv#installation) (get pyenv setup first)

```
# Use a local python that is already installed
pyenv local 3.11.2

# Install a specific version
pyenv install 3.11.2

# Check local version
pyenv version
=> 3.11.2 (set by path/to/doc-chat/.python-version)
```

# Installing

We are using [Poetry](https://github.com/python-poetry/poetry) to manage packages and to run a virtual environment, get that installed first.

Then get packages and env setup via:

## Install the dependencies

```
poetry install
```

## Now start a shell and virtual environment

```
poetry shell
```

# Local environment

You need:

1. A `.env` file (always needed), see `.env.example` for an example

Ask Matt for all the keys, you'll need an OpenAPI Key and the right Pinecone settings, although for now just put anything in because the frontend is hard coded to point at [opio-chat.fly.dev](opio-chat.fly.dev) which will just work.

remember `poetry shell` first

```
make start
```

Then it should run
[http://127.0.0.1:9000](http://127.0.0.1:9000)

# Tests

We use pytest for running all our tests.

 To run them all:

```
pytest
```

Test names should follow a pattern of ```test_[module]_[tested behaviour]```. For example: ```test_trademark_table_parser_loads```

For those with an extra penchant for automation we have included the [pytest-watch](https://github.com/joeyespo/pytest-watch) package which can automatically rerun tests if you just run.

```
ptw
```

in the project root. Productivity 🔥. You're welcome.

If you want you can be quite specific about how you want tests to run and which folders to monitor and which tests to rerun first. Example;

```
ptw -- --last-failed --new-first
```

# Deployment

When you commit to `main` a build is run on [https://app.circleci.com/pipelines/github/opioinc/doc-chat?branch=main](https://app.circleci.com/pipelines/github/opioinc/doc-chat?branch=main) - ask Matt for acccess and then automatically deployed to the glorious [fly.io](www.fly.io) platform. (Again - ask Matt for access) then it gets automatically deployed to [https://opio-chat.fly.dev/](https://opio-chat.fly.dev/) - will set up a proper domain name soon.

It's deployed using a docker container so if youw want to change the deployment process you can do that in the `Dockerfile` in the root.
