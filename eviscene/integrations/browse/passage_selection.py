"""Deterministic evidence passage selection; no network or model calls."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

MAX_EXTRACT_INPUT_CHARS = 24000

DEFAULT_EXTRACT_MAX_PASSAGES = 24


def compress_whitespace(text: str) -> str:
    return " ".join(text.split())


class PassageSelectionMixin:
    """Text-only passage selection, independent of fetch and model transports."""

    @staticmethod
    def _passage_has_unresolved_referent(value: str) -> bool:
        """Detect a passage whose central media/event referent lives upstream."""

        text = " ".join(str(value or "").casefold().split())
        patterns = (
            r"\b(?:this|that|said|such|the\s+aforementioned)\s+"
            r"(?:video|image|photo|post|advertisement|ad|campaign|event|record)\b",
            r"\b(?:video|image|photo|post|advertisement|ad|campaign|event)\s+"
            r"(?:in\s+question|mentioned\s+above)\b",
            r"\b(?:videoclipul|clipul|materialul|imaginea|fotografia|reclama|"
            r"campania|postarea)\s+(?:respectiv(?:ă|a|ul)?|menționat(?:ă|a|ul)?)\b",
            r"\b(?:acest|această|acel|acea)\s+"
            r"(?:videoclip|clip|material|imagine|fotografie|reclamă|campanie|postare)\b",
        )
        return any(re.search(pattern, text, flags=re.UNICODE) for pattern in patterns)

    @staticmethod
    def _preceding_passage(
        passages: List[Dict[str, Any]],
        current: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return the nearest non-empty paragraph before a deictic passage."""

        current_start = int(current.get("start", 0) or 0)
        candidates = [
            passage
            for passage in passages
            if int(passage.get("end", 0) or 0) <= current_start
            and str(passage.get("text", "")).strip()
        ]
        if not candidates:
            return None
        return dict(max(candidates, key=lambda item: int(item.get("end", 0) or 0)))

    @classmethod
    def _select_goal_passages(
        cls,
        passages: List[Dict[str, Any]],
        goal: str,
        *,
        hint: str = "",
        max_chars: int,
        max_passages: int = DEFAULT_EXTRACT_MAX_PASSAGES,
    ) -> List[Dict[str, Any]]:
        """Keep the Claim relation visible while a free-form hint guides ranking."""

        if not passages:
            return []
        goal_tokens = cls._ranking_tokens(goal)
        hint_tokens = cls._ranking_tokens(hint)
        ranked: List[tuple[float, float, int, Dict[str, Any]]] = []
        for index, passage in enumerate(passages):
            goal_score = cls._goal_passage_score(
                passage,
                goal,
                goal_tokens=goal_tokens,
            )
            hint_score = (
                cls._goal_passage_score(
                    passage,
                    hint,
                    goal_tokens=hint_tokens,
                )
                if hint_tokens
                else 0.0
            )
            ranked.append((max(goal_score, hint_score), goal_score, index, passage))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if hint_tokens:
            claim_anchor = max(
                ranked,
                key=lambda item: (item[1], -item[2]),
            )
            if claim_anchor[1] > 0:
                ranked = [
                    claim_anchor,
                    *(item for item in ranked if item[2] != claim_anchor[2]),
                ]

        selected: List[Dict[str, Any]] = []
        used = 0
        max_chars = min(max(1000, int(max_chars)), MAX_EXTRACT_INPUT_CHARS)
        max_passages = max(1, int(max_passages))
        for _score, _goal_score, _index, passage in ranked:
            if len(selected) >= max_passages:
                break
            text = str(passage.get("text", ""))
            cost = len(text) + 32
            if selected and used + cost > max_chars:
                continue
            selected.append(dict(passage))
            used += cost
            if used >= max_chars:
                break
        selected.sort(key=lambda item: int(item.get("start", 0)))
        for passage_id, passage in enumerate(selected):
            passage["passage_id"] = passage_id
        return selected

    @classmethod
    def _goal_passage_score(
        cls,
        passage: Dict[str, Any],
        goal: str,
        *,
        goal_tokens: Optional[set[str]] = None,
    ) -> float:
        text = str(passage.get("text", ""))
        resolved_goal_tokens = (
            goal_tokens if goal_tokens is not None else cls._ranking_tokens(goal)
        )
        tokens = cls._ranking_tokens(text)
        overlap = len(resolved_goal_tokens & tokens)
        coverage = overlap / max(1, len(resolved_goal_tokens))
        density = overlap / max(1, len(tokens))
        phrase_bonus = sum(
            1.0
            for phrase in re.findall(r'"([^"]{3,})"', str(goal or ""))
            if phrase.casefold() in text.casefold()
        )
        return (8.0 * coverage) + (2.0 * density) + (4.0 * phrase_bonus)

    @staticmethod
    def _ranking_tokens(value: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "does",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "this",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "with",
        }
        return {
            token
            for token in re.findall(
                r"[\w]+",
                str(value or "").casefold(),
                flags=re.UNICODE,
            )
            if len(token) > 1 and token not in stopwords
        }

    @classmethod
    def _prepare_evidence_document(cls, content: str) -> str:
        """Build a stable paragraph document for exact evidence offsets.

        Jina responses often start with metadata and large site-navigation blocks.
        Security and prompt-injection checks still inspect the untouched response;
        this normalization only defines the passage-selection document.
        """

        text = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n\s*\n+", text)
        cleaned: List[str] = []
        for raw_block in blocks:
            block = raw_block.strip()
            if not block:
                continue
            if re.match(
                r"^(URL Source|Published Time)\s*:",
                block,
                flags=re.IGNORECASE,
            ):
                continue
            # Jina does not consistently put a blank line after this marker.
            # Remove the marker itself without discarding body text that shares
            # the same block.
            block = re.sub(
                r"^Markdown Content\s*:\s*",
                "",
                block,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            if not block:
                continue
            title_match = re.match(
                r"^Title\s*:\s*(.+)$", block, flags=re.IGNORECASE | re.DOTALL
            )
            if title_match:
                title = compress_whitespace(title_match.group(1))
                if title:
                    cleaned.append(title)
                continue

            original_lines = [
                line.strip() for line in block.splitlines() if line.strip()
            ]
            link_count = len(re.findall(r"(?<!!)\[[^\]]+\]\([^\)]+\)", block))
            image_count = len(re.findall(r"!\[[^\]]*\]\([^\)]+\)", block))
            bullet_lines = sum(
                bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line))
                for line in original_lines
            )
            if cls._is_navigation_block(
                original_lines,
                link_count=link_count,
                image_count=image_count,
                bullet_lines=bullet_lines,
            ):
                continue

            block = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", block)
            block = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", block)
            block = re.sub(r"https?://\S+", " ", block)
            normalized_lines: List[str] = []
            for line in block.splitlines():
                line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
                line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
                line = compress_whitespace(line)
                if line and line not in {"* * *", "---", "Back", "Search"}:
                    normalized_lines.append(line)
            normalized = compress_whitespace(" ".join(normalized_lines))
            if len(normalized) >= 20:
                cleaned.append(normalized)
        return "\n\n".join(cleaned)

    @staticmethod
    def _is_navigation_block(
        lines: List[str],
        *,
        link_count: int,
        image_count: int,
        bullet_lines: int,
    ) -> bool:
        if not lines:
            return True
        compact = compress_whitespace(" ".join(lines)).lower()
        if compact in {
            "explore",
            "search",
            "back",
            "featured",
            "suggested searches",
            "news & events",
            "multimedia",
        }:
            return True
        line_count = len(lines)
        if image_count and image_count >= max(1, line_count - 1):
            return True
        if link_count >= 2 and bullet_lines >= max(1, line_count // 2):
            return True
        if link_count >= 4 and not re.search(r"[.!?]\s", compact):
            return True
        return False

    @staticmethod
    def _build_evidence_passages(
        content: str,
        max_chars: int = 700,
    ) -> List[Dict[str, Any]]:
        passages: List[Dict[str, Any]] = []
        paragraph_pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
        for source_block_id, match in enumerate(paragraph_pattern.finditer(content)):
            paragraph_start, paragraph_end = match.span()
            cursor = paragraph_start
            while cursor < paragraph_end:
                target_end = min(paragraph_end, cursor + max_chars)
                end = target_end
                if target_end < paragraph_end:
                    search_floor = min(target_end, cursor + max_chars // 2)
                    boundaries = [
                        content.rfind(marker, search_floor, target_end)
                        for marker in (". ", "? ", "! ", "; ", "。", "？", "！", "；")
                    ]
                    boundary = max(boundaries)
                    if boundary >= search_floor:
                        end = boundary + 1
                    else:
                        space = content.rfind(" ", search_floor, target_end)
                        if space >= search_floor:
                            end = space
                while end > cursor and content[end - 1].isspace():
                    end -= 1
                if end <= cursor:
                    end = target_end
                passages.append(
                    {
                        "passage_id": len(passages),
                        "start": cursor,
                        "end": end,
                        "text": content[cursor:end],
                        "source_block_id": source_block_id,
                        "source_block_start": paragraph_start,
                        "source_block_end": paragraph_end,
                        "source_block_text": content[paragraph_start:paragraph_end],
                    }
                )
                cursor = end
                while cursor < paragraph_end and content[cursor].isspace():
                    cursor += 1
        return passages

    @staticmethod
    def _format_evidence_passages(passages: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"[PASSAGE {passage['passage_id']}] {passage['text']}"
            for passage in passages
        )

    @staticmethod
    def _context_passages(
        passages: List[Dict[str, Any]],
        current: Mapping[str, Any],
        *,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        """Return complete nearby source chunks without cutting their text.

        The extractor may select a chunk from a long source paragraph.  The
        model-visible record keeps that exact selected text and adds nearby
        complete chunks so pronouns, qualifications, and the paragraph's
        surrounding context are not silently removed.  We only omit whole
        chunks when the bounded context window is full; no selected chunk is
        sliced.
        """

        if not passages:
            return []
        ordered = sorted(
            (dict(item) for item in passages if str(item.get("text", "")).strip()),
            key=lambda item: int(item.get("start", 0) or 0),
        )
        current_start = int(current.get("start", 0) or 0)
        current_index = next(
            (
                index
                for index, item in enumerate(ordered)
                if int(item.get("start", 0) or 0) == current_start
            ),
            -1,
        )
        if current_index < 0:
            return [dict(current)]

        limit = max(1200, int(max_chars))
        current_passage = ordered[current_index]
        source_block_text = str(current_passage.get("source_block_text", "")).strip()
        source_block_start = int(current_passage.get("source_block_start", 0) or 0)
        source_block_end = int(current_passage.get("source_block_end", 0) or 0)
        if source_block_text and len(source_block_text) <= limit:
            selected = [
                {
                    **current_passage,
                    "text": source_block_text,
                    "start": source_block_start,
                    "end": source_block_end,
                    "context_role": "selected_source_paragraph",
                }
            ]
            # A deictic sentence often depends on the preceding paragraph.
            # Include that complete paragraph when it fits, preserving the
            # relationship without cutting either paragraph.
            if PassageSelectionMixin._passage_has_unresolved_referent(
                current_passage.get("text", "")
            ):
                previous = next(
                    (
                        item
                        for item in reversed(ordered[:current_index])
                        if str(item.get("source_block_text", "")).strip()
                        and int(item.get("source_block_id", -1))
                        != int(current_passage.get("source_block_id", -2))
                    ),
                    None,
                )
                if previous is not None:
                    previous_text = str(previous.get("source_block_text", ""))
                    if len(source_block_text) + len(previous_text) + 2 <= limit:
                        selected.insert(
                            0,
                            {
                                **previous,
                                "text": previous_text,
                                "start": int(
                                    previous.get("source_block_start", 0) or 0
                                ),
                                "end": int(previous.get("source_block_end", 0) or 0),
                                "context_role": "referent_context",
                            },
                        )
            return sorted(
                selected,
                key=lambda item: int(item.get("start", 0) or 0),
            )

        # A pathological single source block can exceed the bounded context
        # window. In that case retain complete extractor chunks and never slice
        # the selected chunk itself.
        selected: List[Dict[str, Any]] = [ordered[current_index]]
        used = len(str(selected[0].get("text", "")))
        candidate_indexes: List[int] = []
        for distance in range(1, len(ordered) + 1):
            for index in (current_index - distance, current_index + distance):
                if 0 <= index < len(ordered):
                    candidate_indexes.append(index)
        for index in candidate_indexes:
            candidate = ordered[index]
            cost = len(str(candidate.get("text", ""))) + 2
            if used + cost > limit:
                continue
            selected.append(candidate)
            used += cost
        return sorted(
            selected,
            key=lambda item: int(item.get("start", 0) or 0),
        )

    @staticmethod
    def _format_context_passages(
        passages: List[Dict[str, Any]],
    ) -> str:
        return "\n\n".join(
            str(passage.get("text", ""))
            for passage in passages
            if str(passage.get("text", "")).strip()
        )
