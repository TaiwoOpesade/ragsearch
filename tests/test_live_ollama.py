"""Opt-in smoke tests against a real local Ollama (RAGSEARCH_LIVE=1)."""

import numpy as np
import pytest

from ragsearch.config import load_settings
from ragsearch.embedder import OllamaEmbedder

pytestmark = pytest.mark.live


def test_real_ollama_embeddings_are_semantic():
    settings = load_settings()
    emb = OllamaEmbedder(model=settings.embed_model, host=settings.ollama_host)
    q, near, far = emb.embed(
        ["how do I reverse an undo history", "a stack supports undo", "photosynthesis in leaves"]
    )
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert q @ near > q @ far


def test_real_ollama_answers_with_a_citation():
    from ragsearch.generator import OllamaGenerator, cited_numbers
    from ragsearch.retrieval import Hit

    settings = load_settings()
    gen = OllamaGenerator(model=settings.gen_model, host=settings.ollama_host)
    sources = [Hit(1.0, "stacks.md", "A stack is a last in first out (LIFO) structure.", "local")]
    answer = gen.generate("What order does a stack remove items in?", sources)
    assert cited_numbers(answer, n_sources=1) == [1]
