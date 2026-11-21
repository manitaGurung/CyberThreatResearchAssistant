
import math
import re

import pandas as pd

from research_pipeline.retrieval import (
    hybrid_retrieval,
    rerank_results
)

EVIDENCE_CITATION_PATTERN = re.compile(
    r"\[([^\[\]]*\bE\d+\b[^\[\]]*)\]",
    re.IGNORECASE
)
EVIDENCE_ID_PATTERN = re.compile(r"\bE(\d+)\b", re.IGNORECASE)
NUMBERED_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _metadata_value(value):
    """Return a displayable metadata value, or None when it is unavailable."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None

    return value


def _display_year(value):
    value = _metadata_value(value)
    if value is None:
        return None

    try:
        numeric_year = float(value)
        if numeric_year.is_integer():
            return str(int(numeric_year))
    except ValueError:
        pass

    return value


def _display_page(value):
    value = _metadata_value(value)
    if value is None:
        return None

    try:
        numeric_page = float(value)
        if numeric_page.is_integer():
            return str(int(numeric_page))
    except ValueError:
        pass

    return value


def _reference_identity(passage):
    article_id = _metadata_value(passage.get("article_id"))
    title = _metadata_value(passage.get("title"))
    if article_id:
        identity = ("article_id", article_id.casefold())
    elif title:
        identity = ("title", title.casefold())
    else:
        identity = None

    return identity, article_id, title


def _format_reference(reference):
    title = reference["title"]
    year = reference["year"]
    article_id = reference["article_id"]
    pages = reference["pages"]

    description = title or article_id
    if title and year:
        description = f"{title} ({year})"
    elif year:
        description = f"{year}"

    parts = [f"[{reference['number']}] {description}."]
    if article_id:
        parts.append(f"PDF file: {article_id}.")
    if pages:
        pages = sorted(pages, key=_page_sort_key)
        page_word = "page" if len(pages) == 1 else "pages"
        parts.append(f"PDF {page_word} {', '.join(pages)}.")

    return " ".join(parts)


def _page_sort_key(page):
    try:
        numeric_page = float(page)
        if math.isfinite(numeric_page):
            return 0, numeric_page, page.casefold()
    except ValueError:
        pass

    return 1, 0, page.casefold()


def format_answer_citations(answer, evidence):
    """Replace evidence IDs with validated numbered citations and references."""
    references_by_identity = {}
    references = []

    answer = re.sub(
        r"(?im)^\s*#{1,6}\s*references\s*$[\s\S]*$",
        "",
        answer
    ).rstrip()

    # Gemini must cite evidence markers; raw numbers have no reliable source mapping.
    answer = NUMBERED_CITATION_PATTERN.sub("", answer)

    def replace_evidence_citation(match):
        citation_numbers = []

        for evidence_id in EVIDENCE_ID_PATTERN.findall(match.group(1)):
            evidence_index = int(evidence_id) - 1
            if evidence_index < 0 or evidence_index >= len(evidence):
                continue

            passage = evidence[evidence_index]
            identity, article_id, title = _reference_identity(passage)
            if identity is None:
                continue

            if identity not in references_by_identity:
                reference = {
                    "number": len(references) + 1,
                    "article_id": article_id,
                    "title": title,
                    "year": _display_year(passage.get("year")),
                    "pages": [],
                    "_page_keys": set()
                }
                references_by_identity[identity] = reference
                references.append(reference)
            else:
                reference = references_by_identity[identity]
                reference["article_id"] = reference["article_id"] or article_id
                reference["title"] = reference["title"] or title
                reference["year"] = reference["year"] or _display_year(
                    passage.get("year")
                )

            page = _display_page(passage.get("page_number"))
            if page is not None and page not in reference["_page_keys"]:
                reference["_page_keys"].add(page)
                reference["pages"].append(page)

            if reference["number"] not in citation_numbers:
                citation_numbers.append(reference["number"])

        if not citation_numbers:
            return ""
        return f"[{', '.join(str(number) for number in citation_numbers)}]"

    formatted_answer = EVIDENCE_CITATION_PATTERN.sub(
        replace_evidence_citation,
        answer
    )
    formatted_answer = re.sub(
        r"\[[^\]]*\bE\d+\b[^\]]*(?:\]|$)",
        "",
        formatted_answer,
        flags=re.IGNORECASE
    )

    formatted_references = [
        _format_reference(reference)
        for reference in references
    ]

    reference_numbers = [
        int(number)
        for reference in formatted_references
        for number in re.findall(r"^\[(\d+)\]", reference)
    ]
    citation_numbers = {
        int(number)
        for citation_group in NUMBERED_CITATION_PATTERN.findall(
            formatted_answer
        )
        for number in re.findall(r"\d+", citation_group)
    }

    if (
        re.search(r"\[\s*E\d+", formatted_answer, re.IGNORECASE)
        or len(reference_numbers) != len(set(reference_numbers))
        or set(reference_numbers) != set(range(1, len(references) + 1))
        or not citation_numbers.issubset(set(reference_numbers))
        or len(references_by_identity) != len(references)
    ):
        raise ValueError("Citation and reference mapping validation failed.")

    return formatted_answer, formatted_references


def generate_evidence_grounded_answer(
    query,
    chunks,
    faiss_index,
    embedding_model,
    bm25_index,
    cross_encoder,
    gemini_client,
    top_k=5
):
    """
    Generate an evidence-grounded academic answer using:
    Hybrid Retrieval + Cross-Encoder Reranking + Gemini.
    """

    # 1. Retrieve relevant passages using hybrid retrieval
    candidates = hybrid_retrieval(
        query=query,
        chunks=chunks,
        faiss_index=faiss_index,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        top_k=10,
        candidate_k=20
    )

    # 2. Rerank retrieved passages
    evidence = rerank_results(
        query,
        candidates,
        cross_encoder,
        top_k=top_k
    )

    # 3. Prepare evidence with complete article metadata
    context = ""

    for i, passage in enumerate(evidence, start=1):

        year = passage["year"]

        if pd.notna(year):
            year = int(float(year))

        context += f"""
Evidence {i}
Citation marker: [E{i}]
Article ID: {passage['article_id']}
Article Title: {passage['title']}
Publication Year: {year}
PDF Page Number: {passage['page_number']}
Supporting Passage:
{passage['text']}
"""

    # 4. Prepare the evidence-grounded prompt
    prompt = f"""
You are an academic research assistant.

Answer the research question using only the supplied research evidence.

STRICT REQUIREMENTS:

1. Use only claims supported by the supplied passages.
2. Do not invent algorithms, findings, statistics, or references.
3. Put a citation marker immediately after every factual claim, and cite only
   passage(s) whose supporting text directly establishes that specific claim.
4. Cite supporting passages using only their exact evidence marker, such as [E1].
5. To cite multiple passages, use one bracketed group, such as [E1, E4, E5].
6. Never write numbered citations such as [1] or [2]; only use evidence markers.
7. Do not write a References section or reproduce bibliographic details in the answer.
8. Do not cite a paper merely because it discusses the same broad topic. Match
   every claim to the passage that directly supports it.
9. Keep each individual study's methods and findings distinct from conclusions
   synthesized by systematic reviews or surveys. A review-level count or trend
   must cite the review passage reporting it, not a topical primary study.
10. Compare methods, datasets, or evaluation metrics only when the cited
    passages provide comparable details for both. Otherwise, state that the
    retrieved evidence does not provide sufficient information for comparison.
11. Clearly distinguish directly reported findings from any synthesis across
    sources; cite every source contributing to a synthesis.
12. Do not infer missing study details or combine separate findings as if they
    came from one paper.
13. If evidence is insufficient, state what cannot be established.
14. Use clear academic language.

Research Question:
{query}

Research Evidence:
{context}

Write a structured answer with appropriate citations.
"""

    # 5. Generate the answer using Gemini Flash-Lite
    try:
        interaction = gemini_client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt
        )

        answer = interaction.output_text

    except Exception as e:
        answer = f"Gemini API request failed: {str(e)}"

    answer, references = format_answer_citations(answer, evidence)

    return {
        "question": query,
        "answer": answer,
        "references": references,
        "evidence": evidence
    }
