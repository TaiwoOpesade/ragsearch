"""Pydantic request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Origin = Literal["local", "web"]


class Health(BaseModel):
    status: str
    chunks: int
    embedder: str
    index_backend: str
    web_configured: bool = Field(False, description="whether TAVILY_API_KEY is set")
    web_credits_used: int | None = Field(None, description="Tavily credits used this month")
    web_credits_cap: int = Field(800, description="TAVILY_MONTHLY_CAP")


class IngestRequest(BaseModel):
    path: str | None = Field(None, min_length=1,
                             description="local file or directory on the server")
    url: str | None = Field(None, pattern=r"^https?://\S+$",
                            description="http(s) page or site to ingest")
    crawl: bool = False
    max_pages: int = Field(30, ge=1, le=500)
    max_depth: int = Field(2, ge=0, le=5)
    provider: Literal["local", "tavily"] = "local"

    @model_validator(mode="after")
    def exactly_one_target(self) -> "IngestRequest":
        if (self.path is None) == (self.url is None):
            raise ValueError("give exactly one of 'path' or 'url'")
        return self


class IngestResponse(BaseModel):
    added: int
    updated: list[str]
    skipped: list[str]
    failed: list[str]
    removed: list[str] = Field(default_factory=list,
                               description="indexed files under the folder that no longer exist")
    total_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)
    web: bool = Field(False, description="fall back to Tavily if local results are weak")
    save_web: bool = Field(False, description="add web results to the local index")


class SearchHit(BaseModel):
    score: float
    source: str
    text: str
    origin: Origin


class SearchResponse(BaseModel):
    results: list[SearchHit]
    warnings: list[str] = Field(default_factory=list,
                                description="why the web fallback was skipped or failed")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=20)
    web: bool = False
    save_web: bool = False


class Citation(BaseModel):
    number: int
    source: str = Field(description="file path or URL")
    origin: Origin


class NumberedHit(SearchHit):
    number: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    results: list[NumberedHit]
    warnings: list[str] = Field(default_factory=list)
