import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Tokenize text using the project's original BM25 tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks):
    """Build BM25 index from the sentence-based chunks."""

    tokenized_corpus = [
        tokenize(chunk["text"]) for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 index built successfully.")

    return bm25


def dense_retrieval(query, chunks, faiss_index, embedding_model, top_k=10):
    """Retrieve passages using dense embeddings and FAISS."""

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding, dtype=np.float32
    )

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            **chunks[int(idx)],
            "dense_score": float(score),
            "dense_rank": len(results) + 1
        })

    return results


def bm25_retrieval(query, chunks, bm25_index, top_k=10):
    """Retrieve passages using BM25."""

    tokenized_query = tokenize(query)

    scores = bm25_index.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            **chunks[int(idx)],
            "bm25_score": float(scores[idx]),
            "bm25_rank": len(results) + 1
        })

    return results


def hybrid_retrieval(
    query,
    chunks,
    faiss_index,
    embedding_model,
    bm25_index,
    top_k=5,
    candidate_k=10,
    rrf_k=60
):
    """Combine BM25 and dense retrieval using Reciprocal Rank Fusion."""

    dense_results = dense_retrieval(
        query,
        chunks,
        faiss_index,
        embedding_model,
        top_k=candidate_k
    )

    sparse_results = bm25_retrieval(
        query,
        chunks,
        bm25_index,
        top_k=candidate_k
    )

    fused_scores = {}

    for rank, result in enumerate(dense_results, start=1):
        article_id = result["article_id"]
        page = result["page_number"]
        text = result["text"]

        key = (article_id, page, text)

        fused_scores[key] = fused_scores.get(key, 0) + (
            1 / (rrf_k + rank)
        )

    for rank, result in enumerate(sparse_results, start=1):
        article_id = result["article_id"]
        page = result["page_number"]
        text = result["text"]

        key = (article_id, page, text)

        fused_scores[key] = fused_scores.get(key, 0) + (
            1 / (rrf_k + rank)
        )

    combined = {}

    for result in dense_results + sparse_results:

        key = (
            result["article_id"],
            result["page_number"],
            result["text"]
        )

        if key not in combined:
            combined[key] = result.copy()

        combined[key]["rrf_score"] = fused_scores[key]

    ranked_results = sorted(
        combined.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]


def rerank_results(query, results, cross_encoder, top_k=5):
    """Rerank retrieved passages using a Cross-Encoder."""

    if not results:
        return []

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = cross_encoder.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):

        item = result.copy()
        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]