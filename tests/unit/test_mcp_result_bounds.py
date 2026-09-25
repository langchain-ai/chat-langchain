import json

from connectors.mcp import constrain_search_result


def test_normal_search_result_is_byte_for_byte_unchanged():
    result = '[{"title":"One","link":"/one"}]'

    assert constrain_search_result(result) == result


def test_oversized_search_result_keeps_complete_entries_and_reports_omitted_count():
    entries = [
        {"title": f"Document {index}", "link": f"/docs/{index}"} for index in range(20)
    ]

    result = constrain_search_result(entries, max_bytes=220)
    payload, omitted_line = result.split("\n")
    kept = json.loads(payload)

    assert kept
    assert all(set(entry) == {"title", "link"} for entry in kept)
    assert len(result.encode()) <= 220
    assert omitted_line == f"Omitted {len(entries) - len(kept)} entries."
