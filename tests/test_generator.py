import pytest

from ragsearch.generator import (
    MAX_CONTEXT,
    NO_ANSWER,
    REPLY_TOKENS,
    SYSTEM_PROMPT,
    GenerationError,
    OllamaGenerator,
    build_messages,
    cited_numbers,
)
from ragsearch.retrieval import Hit

HITS = [
    Hit(0.9, "notes/stacks.md", "A stack is last in first out.", "local"),
    Hit(0.7, "https://web.example/queues", "A queue is first in first out.", "web"),
]


def test_prompt_numbers_every_source_with_its_location():
    user = build_messages("What is a stack?", HITS)[-1]["content"]
    assert "[1] (notes/stacks.md)\nA stack is last in first out." in user
    assert "[2] (https://web.example/queues)\nA queue is first in first out." in user
    assert user.rstrip().endswith("Question: What is a stack?")


def test_system_prompt_restricts_answers_to_sources_and_requires_citations():
    system = build_messages("q", HITS)[0]
    assert system["role"] == "system"
    text = system["content"]
    assert "only" in text
    assert "[1]" in text and "[2]" in text
    assert NO_ANSWER in text


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Stacks are LIFO [1].", [1]),
        ("Both [2] and [1], again [1].", [1, 2]),
        ("See [1, 2].", [1, 2]),
        ("No citations here.", []),
        ("Out of range [7] ignored [2].", [2]),
    ],
)
def test_cited_numbers_extracts_valid_citations(answer, expected):
    assert cited_numbers(answer, n_sources=2) == expected


# --- OllamaGenerator against a local stub server ---------------------------


def chat_ok(request):
    return 200, {}, {"model": request["json"]["model"],
                     "message": {"role": "assistant", "content": "A stack is LIFO [1]."},
                     "done": True}


def test_ollama_generator_posts_non_streaming_chat(stub_server):
    stub_server.routes["/api/chat"] = chat_ok
    gen = OllamaGenerator(model="llama3.2:3b", host=stub_server.url)
    assert gen.generate("What is a stack?", HITS) == "A stack is LIFO [1]."
    body = stub_server.requests[0]["json"]
    assert body["model"] == "llama3.2:3b"
    assert body["stream"] is False  # /api/chat streams unless told not to
    assert body["messages"] == build_messages("What is a stack?", HITS)


def test_ollama_generator_reports_server_errors(stub_server):
    stub_server.routes["/api/chat"] = (404, {}, {"error": "model 'nope' not found"})
    with pytest.raises(GenerationError, match="not found"):
        OllamaGenerator(model="nope", host=stub_server.url).generate("q", HITS)


def test_ollama_generator_unreachable_is_a_clear_error():
    gen = OllamaGenerator(host="http://127.0.0.1:1", timeout=1)
    with pytest.raises(GenerationError, match="Ollama"):
        gen.generate("q", HITS)


# --- context sizing and unreadable replies ---------------------------------------


def many_hits(n, chars):
    return [Hit(1.0 - i / 100, f"notes/{i}.md", "word " * (chars // 5), "local") for i in range(n)]


def test_small_prompt_uses_the_default_context(stub_server):
    stub_server.routes["/api/chat"] = chat_ok
    OllamaGenerator(host=stub_server.url).generate("q", HITS)
    assert stub_server.requests[0]["json"]["options"]["num_ctx"] == 4096


def test_context_grows_so_twenty_sources_are_not_truncated(stub_server):
    stub_server.routes["/api/chat"] = chat_ok
    OllamaGenerator(host=stub_server.url).generate("q", many_hits(20, 1300))
    body = stub_server.requests[0]["json"]
    prompt_chars = sum(len(m["content"]) for m in body["messages"])
    num_ctx = body["options"]["num_ctx"]
    assert num_ctx > 4096
    assert num_ctx * 3 >= prompt_chars  # room for the whole prompt at ~3 chars/token
    assert num_ctx % 1024 == 0


def test_sources_that_cannot_fit_are_dropped_from_the_end(stub_server):
    stub_server.routes["/api/chat"] = chat_ok
    OllamaGenerator(host=stub_server.url).generate("q", many_hits(60, 2000))
    body = stub_server.requests[0]["json"]
    user = body["messages"][-1]["content"]
    assert body["options"]["num_ctx"] <= MAX_CONTEXT
    assert "[1] (notes/0.md)" in user  # the best sources and the rules always survive
    assert "[60]" not in user
    assert body["messages"][0]["content"] == SYSTEM_PROMPT


@pytest.mark.parametrize("n_sources", [5, 20])
def test_cjk_sources_get_a_context_big_enough_for_their_characters(stub_server, n_sources):
    # Chinese, Japanese and Korean run at about one token per character and have no
    # spaces, so one 200-word chunk can hold thousands of characters.
    stub_server.routes["/api/chat"] = chat_ok
    chunk = "栈是后进先出的数据结构。" * 160  # "A stack is a LIFO data structure.", 1,920 chars
    hits = [Hit(1.0 - i / 100, f"notes/{i}.md", chunk, "local") for i in range(n_sources)]
    OllamaGenerator(host=stub_server.url).generate("栈是什么？", hits)  # "What is a stack?"
    body = stub_server.requests[0]["json"]
    cjk_chars = sum(not c.isascii() for m in body["messages"] for c in m["content"])
    assert body["options"]["num_ctx"] >= cjk_chars + REPLY_TOKENS  # weaker sources dropped
    assert "[1] (notes/0.md)" in body["messages"][-1]["content"]


def test_unreadable_reply_is_a_generation_error(stub_server):
    stub_server.routes["/api/chat"] = (200, {"Content-Type": "text/html"}, "<html></html>")
    with pytest.raises(GenerationError, match="unreadable"):
        OllamaGenerator(host=stub_server.url).generate("q", HITS)


def test_error_field_in_a_200_reply_is_a_generation_error(stub_server):
    stub_server.routes["/api/chat"] = (200, {}, {"error": "model ran out of memory"})
    with pytest.raises(GenerationError, match="out of memory"):
        OllamaGenerator(host=stub_server.url).generate("q", HITS)
