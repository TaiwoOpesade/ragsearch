"""Answer questions from retrieved sources with a local LLM (Ollama ``POST /api/chat``).

The prompt numbers each source, tells the model to answer *only* from them,
to cite as [1], [2], and to say it doesn't know when the sources are not enough.
"""

import re
from typing import Protocol

import httpx

from ragsearch.retrieval import Hit

NO_ANSWER = "I don't know based on the provided sources."

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided by the user. "
    "Do not use any outside knowledge. Cite the source of every claim with its number "
    "in square brackets, for example [1] or [2]. If the sources do not contain the "
    f'answer, reply exactly: "{NO_ANSWER}"'
)

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class GenerationError(RuntimeError):
    """Raised when the language model cannot produce an answer."""


class Generator(Protocol):
    def generate(self, question: str, sources: list[Hit]) -> str: ...


def build_messages(question: str, sources: list[Hit]) -> list[dict]:
    blocks = [f"[{n}] ({hit.source})\n{hit.text}" for n, hit in enumerate(sources, start=1)]
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    """Sorted, de-duplicated source numbers cited in ``answer`` (out-of-range ignored)."""
    numbers = {
        int(n) for group in _CITATION.findall(answer) for n in group.split(",")
    }
    return sorted(n for n in numbers if 1 <= n <= n_sources)


class OllamaGenerator:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        timeout: float = 120,
        num_ctx: int = 4096,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx  # fits a 3B model plus five 200-word chunks in ~6GB VRAM

    def generate(self, question: str, sources: list[Hit]) -> str:
        body = {
            "model": self.model,
            "messages": build_messages(question, sources),
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": self.num_ctx},
        }
        try:
            response = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        except httpx.HTTPError as err:
            raise GenerationError(
                f"Could not reach Ollama at {self.host}. Is it running? ({err})"
            ) from err
        if response.status_code != 200:
            try:
                message = response.json().get("error", response.text)
            except ValueError:
                message = response.text
            raise GenerationError(f"Ollama returned HTTP {response.status_code}: {message}")
        return response.json()["message"]["content"].strip()
