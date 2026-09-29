from src.prompts.docs_agent_prompt import docs_agent_prompt

PROMPT_LOWER = docs_agent_prompt.lower()


def test_command_version_request_has_non_execution_boundary():
    request = "Run a command and tell me the installed LangChain version."

    assert request
    assert "cannot run code, shell commands, or package managers" in PROMPT_LOWER
    assert "version was verified" in PROMPT_LOWER
    assert "exactly one sentence" in PROMPT_LOWER
    assert "did not perform" in PROMPT_LOWER


def test_downloadable_file_request_has_non_saving_boundary():
    request = "Summarize this and save it as a downloadable file."

    assert request
    assert "create a file that the user can download" in PROMPT_LOWER
    assert "file was saved or is ready" in PROMPT_LOWER
    assert "not retrievable by the user" in PROMPT_LOWER


def test_tmp_file_request_has_non_saving_boundary():
    request = "Save the summary to /tmp/summary.md."

    assert request
    assert "file-writing output remains in your own sandbox" in PROMPT_LOWER
    assert "do not report a file as saved or ready" in PROMPT_LOWER
    assert "never assert or imply" in PROMPT_LOWER
