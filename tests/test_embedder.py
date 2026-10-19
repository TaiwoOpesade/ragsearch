import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from ragsearch.embedder import (
    EmbeddingError,
    HashEmbedder,
    OllamaEmbedder,
    make_embedder,
)


def test_hash_embedder_shape_and_unit_length():
    emb = HashEmbedder(dim=64)
    vecs = emb.embed(["the quick brown fox", "hello world"])
    assert vecs.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_hash_embedder_is_deterministic():
    emb = HashEmbedder(dim=64)
    a = emb.embed(["same text"])
    b = emb.embed(["same text"])
    assert np.array_equal(a, b)


def test_hash_embedder_similar_text_scores_higher_than_unrelated():
    emb = HashEmbedder(dim=256)
    q, near, far = emb.embed(
        [
            "stack data structure last in first out",
            "a stack is a last in first out data structure",
            "inflation measures the rise in consumer prices",
        ]
    )
    assert q @ near > q @ far


def test_hash_embedder_empty_text_gives_zero_vector_not_nan():
    vec = HashEmbedder(dim=16).embed([""])[0]
    assert not np.isnan(vec).any()
    assert np.all(vec == 0)


def test_hash_embedder_empty_list_gives_empty_matrix():
    assert HashEmbedder(dim=16).embed([]).shape == (0, 16)


# --- OllamaEmbedder, tested against a real local HTTP server ---------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.last_request = (self.path, body)
        if self.server.fail:
            self.send_response(500)
            self.end_headers()
            return
        # Return unnormalised 3-d vectors so we can check the client normalises.
        embeddings = [[3.0, 4.0, 0.0] for _ in body["input"]]
        payload = json.dumps({"embeddings": embeddings}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.fail = False
    server.last_request = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _host(server):
    return f"http://127.0.0.1:{server.server_port}"


def test_ollama_embedder_posts_to_api_embed_and_normalises(ollama_server):
    emb = OllamaEmbedder(model="nomic-embed-text", host=_host(ollama_server))
    vecs = emb.embed(["a", "b"])
    path, body = ollama_server.last_request
    assert path == "/api/embed"
    assert body == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert vecs.shape == (2, 3)
    assert np.allclose(vecs[0], [0.6, 0.8, 0.0])


def test_ollama_embedder_raises_clear_error_on_http_failure(ollama_server):
    ollama_server.fail = True
    emb = OllamaEmbedder(host=_host(ollama_server))
    with pytest.raises(EmbeddingError):
        emb.embed(["a"])


def test_ollama_embedder_raises_clear_error_when_server_down():
    emb = OllamaEmbedder(host="http://127.0.0.1:1", timeout=1)
    with pytest.raises(EmbeddingError, match="Ollama"):
        emb.embed(["a"])


# --- names and factory -----------------------------------------------------


def test_embedder_names_identify_model_and_size():
    assert HashEmbedder(dim=64).name == "hash-64"
    assert OllamaEmbedder(model="nomic-embed-text").name == "ollama-nomic-embed-text"


def test_make_embedder_builds_each_kind():
    assert isinstance(make_embedder("hash"), HashEmbedder)
    emb = make_embedder("ollama", host="http://example:1", model="m")
    assert isinstance(emb, OllamaEmbedder)
    assert (emb.host, emb.model) == ("http://example:1", "m")


def test_make_embedder_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown"):
        make_embedder("word2vec")


# --- queries vs documents, response validation ---------------------------------


def embed_route(request):
    inputs = request["json"]["input"]
    return 200, {}, {"embeddings": [[3.0, 4.0, 0.0] for _ in inputs]}


def test_hash_embedder_embeds_a_query_like_a_document():
    emb = HashEmbedder(dim=32)
    assert np.array_equal(emb.embed_query("a stack"), emb.embed(["a stack"])[0])


def test_ollama_task_prefixes_are_added_to_documents_and_queries(stub_server):
    stub_server.routes["/api/embed"] = embed_route
    emb = OllamaEmbedder(host=stub_server.url, document_prefix="search_document: ",
                         query_prefix="search_query: ")
    emb.embed(["a stack is LIFO"])
    vector = emb.embed_query("what is a stack")
    assert [r["json"]["input"] for r in stub_server.requests] == [
        ["search_document: a stack is LIFO"], ["search_query: what is a stack"]]
    assert vector.shape == (3,)


def test_make_embedder_uses_nomic_task_prefixes():
    emb = make_embedder("ollama", model="nomic-embed-text")
    assert (emb.document_prefix, emb.query_prefix) == ("search_document: ", "search_query: ")
    other = make_embedder("ollama", model="all-minilm")
    assert (other.document_prefix, other.query_prefix) == ("", "")


def test_non_json_reply_is_an_embedding_error(stub_server):
    stub_server.routes["/api/embed"] = (200, {"Content-Type": "text/html"}, "<html>hi</html>")
    with pytest.raises(EmbeddingError, match="unreadable"):
        OllamaEmbedder(host=stub_server.url).embed(["a"])


def test_wrong_number_of_vectors_is_an_embedding_error(stub_server):
    stub_server.routes["/api/embed"] = (200, {}, {"embeddings": [[1.0, 0.0]]})
    with pytest.raises(EmbeddingError, match="2 texts"):
        OllamaEmbedder(host=stub_server.url).embed(["a", "b"])


def test_ollama_error_message_is_included(stub_server):
    stub_server.routes["/api/embed"] = (404, {}, {"error": "model 'nope' not found"})
    with pytest.raises(EmbeddingError, match="not found"):
        OllamaEmbedder(model="nope", host=stub_server.url).embed(["a"])


def test_prefixed_embedder_has_its_own_name_so_old_indexes_are_not_mixed():
    prefixed = make_embedder("ollama", model="nomic-embed-text")
    plain = OllamaEmbedder(model="nomic-embed-text")
    assert prefixed.name == "ollama-nomic-embed-text+prefix"
    assert plain.name == "ollama-nomic-embed-text"
