from datetime import datetime
from pydantic import BaseModel, Field
from ai.schemas import Source


class ResearchSession(BaseModel):
    id: int | None = None
    question: str
    answer: str
    sources_used: list[str] = Field(default_factory=list)
    sources_failed: list[str] = Field(default_factory=list)
    duration_ms: int
    created_at: datetime | None = None


class SourceBundle(BaseModel):
    """Orkestratorun nəticəsi: gələn mənbələr + kimin düşdüyü."""
    sources: list[Source]
    used: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)