import pytest

from ragsearch.generator import (
    NO_ANSWER,
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
