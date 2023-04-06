# right_founder

# Python Setup

Install the correct version of python using [pyenv](https://github.com/pyenv/pyenv#installation) (get pyenv setup first)

```
# Use a local python that is already installed
pyenv local 3.11.2

# Install a specific version
pyenv install 3.11.2

# Check local version
pyenv version
=> 3.11.2 (set by path/to/right_founder/.python-version)
```

```

# Installing

We are using [Poetry](https://github.com/python-poetry/poetry) to manage packages and to run a virtual environment, get that installed first.

Then get packages and env setup via:

```

# Install the dependencies

poetry install

# Now start a shell and virtual environment

```
poetry shell
```

# Local environment

You need:

1. A `.env` file (always needed)

Ask Matt

# remember `poetry shell` first

poetry run python  manage.py runserver 8001

```
Then it should run
[http://127.0.0.1:8001](http://127.0.0.1:8001)

# Running livereload
Uncomment the lines in base.py that say:
'# "livereload",
'# "livereload.middleware.LiveReloadScript",


The in a seperate terminal run
```poetry run python manage.py livereload```

# Tests
Before we committed a line of code we wrote tests. True story. We use pytest for running all our tests.

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

If you need to add test HTML data, firstly you add this into your local database using django admin, looking for "Insight html fragments" then you will need to ensure that data is saved to the local file so it can then be migrated in the various environments and on heroku (and of course loaded in tests and in circle ci) do this by generating a new file 

```

poetry run python manage.py dumpdata survey_responses.InsightHTMLFragment -o survey_responses/fixtures/InsightHTMLFragment.json

```

 to do this on flyio in production run:
 ```

flyctl ssh console --app right-founder-staging

# wait for the shell to launch then

/venv/bin/poetry run python manage.py loaddata InsightHTMLFragment.json
```
