#context_builder.py
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

from app.retrieval.context.context_models import (
    BuiltContext,
    ContextItem,
    ContextRequest,
    ContextSection,
    ContextSource,
)


class ContextBuilder:
    """
    Builds bounded, provenance-preserving, generation-ready context.

    Responsibilities:
    - deduplicate retrieval evidence
    - prioritize reranked evidence
    - preserve source diversity
    - preserve graph evidence
    - enforce item and character budgets
    - preserve provenance
    - preserve metadata when requested
    - deterministically trim oversized evidence
    - construct generation-ready context

    This class does not call an LLM and does not generate answers.
    """

    SOURCE_PRIORITY = {
        ContextSource.RERANKED: 5,
        ContextSource.HYBRID: 4,
        ContextSource.GRAPH: 4,
        ContextSource.DENSE: 3,
        ContextSource.SPARSE: 2,
    }

    def build(self, request: ContextRequest) -> BuiltContext:
        """
        Build structured, bounded context from retrieval results.
        """

        if not isinstance(request, ContextRequest):
            raise TypeError("request must be a ContextRequest")

        candidates = list(request.results)

        selected = self._select_items(
            request=request,
            candidates=candidates,
        )

        sections = self._build_sections(
            query=request.query,
            selected_items=selected,
            include_metadata=request.include_metadata,
            include_provenance=request.include_provenance,
        )

        character_count = sum(
            len(section.content)
            for section in sections
        )

        return BuiltContext(
            query=request.query,
            sections=sections,
            selected_items=selected,
            total_candidates=len(candidates),
            selected_count=len(selected),
            character_count=character_count,
            truncated=len(selected) < len(candidates)
            or any(
                len(item.text) != len(original.text)
                for item in selected
                for original in candidates
                if item.id == original.id
            ),
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_items(
        self,
        request: ContextRequest,
        candidates: list[ContextItem],
    ) -> list[ContextItem]:
        """
        Select evidence while respecting the actual generation context
        character budget.

        Selection happens against the formatted evidence representation,
        not merely raw document text.
        """

        deduplicated = self._deduplicate(candidates)

        ordered = sorted(
            deduplicated,
            key=self._priority_key,
        )

        selected: list[ContextItem] = []

        for candidate in self._ordered_candidates(
            ordered,
        ):
            if len(selected) >= request.budget.max_items:
                break

            remaining_budget = self._remaining_evidence_budget(
                request=request,
                selected=selected,
            )

            if remaining_budget <= 0:
                break

            prepared = self._fit_item_to_budget(
                item=candidate,
                remaining_budget=remaining_budget,
                include_metadata=request.include_metadata,
                include_provenance=request.include_provenance,
            )

            if prepared is None:
                continue

            selected.append(prepared)

        return self._normalize_ranks(selected)

    def _ordered_candidates(
        self,
        candidates: list[ContextItem],
    ) -> list[ContextItem]:
        """
        Preserve source diversity while retaining deterministic ranking.
        """

        selected_ids: set[str] = set()
        ordered_candidates: list[ContextItem] = []

        # First pass: one strong candidate from each available source.
        for source in self._preferred_sources(candidates):
            candidate = next(
                (
                    item
                    for item in candidates
                    if item.source == source
                    and item.id not in selected_ids
                ),
                None,
            )

            if candidate is not None:
                ordered_candidates.append(candidate)
                selected_ids.add(candidate.id)

        # Second pass: global ranking.
        for candidate in candidates:
            if candidate.id in selected_ids:
                continue

            ordered_candidates.append(candidate)
            selected_ids.add(candidate.id)

        return ordered_candidates

    def _remaining_evidence_budget(
        self,
        request: ContextRequest,
        selected: list[ContextItem],
    ) -> int:
        """
        Calculate how many characters remain for evidence after accounting
        for the user-query section and already-selected evidence.
        """

        query_section = ContextSection(
            title="User Query",
            content=request.query,
        )

        used = len(query_section.content)

        if selected:
            evidence_content = self._format_evidence(
                selected_items=selected,
                include_metadata=request.include_metadata,
                include_provenance=request.include_provenance,
            )

            used += len(evidence_content)

        return max(
            request.budget.max_characters - used,
            0,
        )

    def _fit_item_to_budget(
        self,
        item: ContextItem,
        remaining_budget: int,
        include_metadata: bool,
        include_provenance: bool,
    ) -> ContextItem | None:
        """
        Fit one evidence item into the remaining generation budget.

        If the complete item does not fit, deterministically trim only the
        document text while preserving provenance and metadata.
        """

        candidate = deepcopy(item)

        full_line = self._format_single_evidence(
            index=1,
            item=candidate,
            include_metadata=include_metadata,
            include_provenance=include_provenance,
        )

        if len(full_line) <= remaining_budget:
            return candidate

        # Determine fixed formatting overhead.
        empty_text_item = deepcopy(candidate)
        empty_text_item.text = ""

        overhead = len(
            self._format_single_evidence(
                index=1,
                item=empty_text_item,
                include_metadata=include_metadata,
                include_provenance=include_provenance,
            )
        )

        available_text = remaining_budget - overhead

        if available_text <= 0:
            return None

        candidate.text = candidate.text[:available_text].rstrip()

        if not candidate.text:
            return None

        final_line = self._format_single_evidence(
            index=1,
            item=candidate,
            include_metadata=include_metadata,
            include_provenance=include_provenance,
        )

        if len(final_line) > remaining_budget:
            return None

        return candidate

    # ------------------------------------------------------------------
    # Deduplication / ranking
    # ------------------------------------------------------------------

    def _deduplicate(
        self,
        candidates: list[ContextItem],
    ) -> list[ContextItem]:
        """
        Deduplicate by document/result ID.

        The highest-priority occurrence is retained.
        """

        ordered = sorted(
            candidates,
            key=self._priority_key,
        )

        unique: OrderedDict[str, ContextItem] = OrderedDict()

        for item in ordered:
            if item.id not in unique:
                unique[item.id] = deepcopy(item)

        return list(unique.values())

    def _priority_key(
        self,
        item: ContextItem,
    ) -> tuple:
        """
        Priority ordering:

        1. source importance
        2. retrieval score
        3. original rank
        4. deterministic ID
        """

        source_priority = self.SOURCE_PRIORITY.get(
            item.source,
            0,
        )

        score = (
            item.score
            if item.score is not None
            else float("-inf")
        )

        return (
            -source_priority,
            -score,
            item.rank,
            item.id,
        )

    def _preferred_sources(
        self,
        candidates: list[ContextItem],
    ) -> list[ContextSource]:
        """
        Determine source order dynamically.
        """

        available = {
            item.source
            for item in candidates
        }

        return [
            source
            for source in (
                ContextSource.RERANKED,
                ContextSource.GRAPH,
                ContextSource.HYBRID,
                ContextSource.DENSE,
                ContextSource.SPARSE,
            )
            if source in available
        ]
    @staticmethod
    def _normalize_ranks(
        items: list[ContextItem],
    ) -> list[ContextItem]:
        """
        Assign deterministic 1-based ranks according to the final
        context ordering.

        ContextBuilder may reorder candidates for source diversity,
        priority, deduplication, and budget constraints. Therefore,
        the rank exposed to downstream consumers must represent the
        final context order.
        """

        normalized: list[ContextItem] = []

        for rank, item in enumerate(
            items,
            start=1,
        ):
            normalized.append(
                item.model_copy(
                    update={
                        "rank": rank,
                    }
                )
            )

        return normalized
    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        query: str,
        selected_items: list[ContextItem],
        include_metadata: bool,
        include_provenance: bool,
    ) -> list[ContextSection]:
        """
        Convert selected evidence into explicit context sections.
        """

        sections: list[ContextSection] = [
            ContextSection(
                title="User Query",
                content=query,
            )
        ]

        if not selected_items:
            sections.append(
                ContextSection(
                    title="Evidence",
                    content="No supporting evidence was retrieved.",
                )
            )

            return sections

        sections.append(
            ContextSection(
                title="Evidence",
                content=self._format_evidence(
                    selected_items=selected_items,
                    include_metadata=include_metadata,
                    include_provenance=include_provenance,
                ),
            )
        )

        return sections

    def _format_evidence(
        self,
        selected_items: list[ContextItem],
        include_metadata: bool,
        include_provenance: bool,
    ) -> str:
        """
        Format all evidence deterministically.
        """

        evidence_lines = [
            self._format_single_evidence(
                index=index,
                item=item,
                include_metadata=include_metadata,
                include_provenance=include_provenance,
            )
            for index, item in enumerate(
                selected_items,
                start=1,
            )
        ]

        return "\n\n".join(evidence_lines)

    def _format_single_evidence(
        self,
        index: int,
        item: ContextItem,
        include_metadata: bool,
        include_provenance: bool,
    ) -> str:
        """
        Format a single evidence item.
        """

        line = f"[Evidence {index}] {item.text}"

        if include_provenance:
            line += (
                f"\nSource: {item.source.value}"
                f"\nRank: {item.rank}"
            )

        if include_metadata and item.metadata:
            metadata = ", ".join(
                f"{key}={value}"
                for key, value in sorted(
                    item.metadata.items(),
                    key=lambda pair: str(pair[0]),
                )
            )

            line += f"\nMetadata: {metadata}"

        return line