
import pandas as pd

from research_pipeline.retrieval import (
    hybrid_retrieval,
    rerank_results
)


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
3. Every factual research claim must have a citation.
4. Cite each source separately. Never combine multiple articles inside one citation.
5. Follow this exact citation format:

[Article ID: filename, Title: article title, Year: year, PDF Page: page number]

6. Copy article IDs, titles, years, and page numbers exactly from the evidence.
7. Do not describe a result from a review as a general result for all studies.
8. Clearly distinguish findings reported by individual papers from broader synthesis.
9. If evidence is insufficient, state what cannot be established.
10. Use clear academic language.

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

    return {
        "question": query,
        "answer": answer,
        "evidence": evidence
    }
