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


# Context sizing. Ollama silently drops the *start* of a prompt that is longer than
# num_ctx, which would lose the grounding rules and the best sources, so the
# context is sized to the prompt instead.
CHARS_PER_TOKEN = 3  # conservative: English averages ~4; paths and URLs fewer
REPLY_TOKENS = 1024  # room left for the answer
MAX_CONTEXT = 16384  # a 3B model's KV cache at this size is ~1.8 GB: fits 6 GB VRAM


def _tokens_needed(messages: list[dict]) -> int:
    chars = sum(len(m["content"]) for m in messages)
    return chars // CHARS_PER_TOKEN + 16 * len(messages) + REPLY_TOKENS


class OllamaGenerator:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120,
        num_ctx: int = 4096,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx  # minimum context; grown to fit longer prompts

    def _fit(self, question: str, sources: list[Hit]) -> tuple[list[dict], int]:
        """Messages that fit in MAX_CONTEXT (dropping the weakest sources) and num_ctx."""
        messages = build_messages(question, sources)
        while len(sources) > 1 and _tokens_needed(messages) > MAX_CONTEXT:
            sources = sources[:-1]
            messages = build_messages(question, sources)
        needed = -(-_tokens_needed(messages) // 1024) * 1024  # round up to 1024
        return messages, min(MAX_CONTEXT, max(self.num_ctx, needed))

    def generate(self, question: str, sources: list[Hit]) -> str:
        messages, num_ctx = self._fit(question, sources)
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": num_ctx},
        }
        try:
            response = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        except httpx.HTTPError as err:
            raise GenerationError(
                f"Could not reach Ollama at {self.host}. Is it running? ({err})"
            ) from err
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200:
            message = payload.get("error") if isinstance(payload, dict) else response.text
            raise GenerationError(f"Ollama returned HTTP {response.status_code}: {message}")
        if isinstance(payload, dict) and payload.get("error"):
            raise GenerationError(f"Ollama could not answer: {payload['error']}")
        try:
            return payload["message"]["content"].strip()
        except (TypeError, KeyError, AttributeError) as err:
            raise GenerationError(
                f"Ollama at {self.host} sent an unreadable reply; is OLLAMA_HOST right?"
            ) from err
