from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.reply_language_middleware import (
    MARKER,
    add_language_note,
    detect_reply_language,
)


def test_detects_short_mixed_chinese():
    assert detect_reply_language("graph.astream 如何用") == "Chinese"
    assert detect_reply_language("entity 相关的内容") == "Chinese"
    assert detect_reply_language("with_fallbacks怎么使用") == "Chinese"


def test_detects_other_scripts():
    assert detect_reply_language("あなたのアーキテクチャを説明して") == "Japanese"
    assert detect_reply_language("체크포인터는 어떻게 쓰나요?").startswith("Korean")
    assert "Cyrillic" in detect_reply_language("как использовать checkpointer")
    assert "Arabic" in detect_reply_language("كيف أستخدم DirectoryLoader")


def test_english_and_latin_get_no_note():
    assert detect_reply_language("how do I use checkpointer?") is None
    assert detect_reply_language("como usar o checkpointer?") is None
    assert detect_reply_language("google") is None


def test_ignores_page_note_and_code():
    q = "fix this ```print('Слайд привет мир')``` please --- Note: The user is viewing 页面"
    assert detect_reply_language(q) is None


def test_note_added_to_last_human_only_and_once():
    msgs = [HumanMessage("hello"), AIMessage("hi"), HumanMessage("如果想再加上 store，该怎么写？")]
    out = add_language_note(msgs)
    assert out is not msgs
    assert MARKER in out[-1].content and "Chinese" in out[-1].content
    assert out[0].content == "hello"
    assert msgs[-1].content == "如果想再加上 store，该怎么写？"  # original untouched
    assert add_language_note(out) is out


def test_latest_english_message_after_chinese_gets_none():
    msgs = [HumanMessage("如何使用"), AIMessage("..."), HumanMessage("thanks, show python code")]
    assert add_language_note(msgs) is msgs


def test_list_content():
    msgs = [HumanMessage(content=[{"type": "text", "text": "意图识别"}])]
    out = add_language_note(msgs)
    assert out[0].content[-1]["text"].startswith("\n\n" + MARKER)
