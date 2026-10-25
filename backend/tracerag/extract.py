"""Entity extraction over text using sliding windows with zero-shot / rule fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterator, List

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedEntity:
    text: str
    type: str
    score: float
    start: int
    end: int
    source: str


@dataclass(frozen=True)
class WordWindow:
    text: str
    offset: int


def sliding_window_words(
    text: str,
    window: int = config.GLINER_WINDOW_WORDS,
    overlap: int = config.GLINER_WINDOW_OVERLAP,
) -> Iterator[WordWindow]:
    """Yields overlapping word windows tagged with their char offset."""
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")

    tokens = [(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return

    step = window - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i : i + window]
        if not chunk:
            break
        start_char = chunk[0][1]
        last_word, last_start = chunk[-1]
        end_char = last_start + len(last_word)
        yield WordWindow(text=text[start_char:end_char], offset=start_char)
        if i + window >= len(tokens):
            break


class EntityExtractor:
    """Extracts entities from text using GLiNER if available, or regex rules."""

    def __init__(self) -> None:
        self._gliner = None

    def _get_gliner(self):
        if self._gliner is None:
            try:
                from gliner import GLiNER
                self._gliner = GLiNER.from_pretrained(config.GLINER_MODEL)
            except Exception as e:
                logger.debug("GLiNER unavailable (%s), using rule-based extractor", e)
                self._gliner = False
        return self._gliner

    def extract(self, text: str) -> List[ExtractedEntity]:
        """Extracts named entities (PR, Ticket, Service, Person, Tool, etc.)."""
        if not text or len(text.strip()) < config.MIN_ENTITY_CHARS:
            return []

        gliner = self._get_gliner()
        if gliner:
            entities = []
            for win in sliding_window_words(text):
                try:
                    preds = gliner.predict_entities(
                        win.text, config.ENTITY_LABELS, threshold=config.GLINER_THRESHOLD
                    )
                    for p in preds:
                        entities.append(ExtractedEntity(
                            text=p["text"],
                            type=p["label"],
                            score=float(p["score"]),
                            start=win.offset + p["start"],
                            end=win.offset + p["end"],
                            source="gliner"
                        ))
                except Exception:
                    continue
            if entities:
                return entities

        # Fast Regex rule-based extraction fallback
        return self._extract_rules(text)

    def _extract_rules(self, text: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        
        # PR matches (#123 or PR #123)
        for m in re.finditer(r"(?:PR\s*#?|#)(\d+)\b", text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                text=m.group(0),
                type="PR",
                score=0.95,
                start=m.start(),
                end=m.end(),
                source="regex"
            ))

        # Ticket / Issue matches (JIRA-123, GH-45)
        for m in re.finditer(r"\b[A-Z]{2,10}-\d+\b", text):
            entities.append(ExtractedEntity(
                text=m.group(0),
                type="Ticket",
                score=0.95,
                start=m.start(),
                end=m.end(),
                source="regex"
            ))

        # Service / Component CamelCase matches (PaymentService, AuthRouter)
        for m in re.finditer(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+(?:Service|Client|Manager|Router|Store|Worker|DB)?\b", text):
            entities.append(ExtractedEntity(
                text=m.group(0),
                type="Service",
                score=0.85,
                start=m.start(),
                end=m.end(),
                source="regex"
            ))

        return entities
