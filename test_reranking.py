
from research_pipeline.data_loader import load_research_data
from research_pipeline.retrieval import (
    build_bm25_index,
    hybrid_retrieval,
    rerank_results
)
from sentence_transformers import SentenceTransformer, CrossEncoder


# Load research data
chunks, index = load_research_data()

# Build BM25
print("\nBuilding BM25 index...")
bm25 = build_bm25_index(chunks)

# Load embedding model
print("\nLoading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Cross-Encoder
print("\nLoading Cross-Encoder...")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Research question
query = "What machine learning techniques are used for phishing website detection?"

# Hybrid retrieval
print("\nPerforming hybrid retrieval...")
hybrid_results = hybrid_retrieval(
    query,
    chunks,
    index,
    embedding_model,
    bm25,
    top_k=10,
    candidate_k=20
)

print("Hybrid candidates:", len(hybrid_results))

# Cross-Encoder reranking
print("\nPerforming Cross-Encoder reranking...")
reranked_results = rerank_results(
    query,
    hybrid_results,
    cross_encoder,
    top_k=5
)

# Display results
print("\nTOP 5 RERANKED EVIDENCE PASSAGES")

for i, result in enumerate(reranked_results, start=1):
    print(f"\n{i}. {result['title']}")
    print(f"   Article: {result['article_id']}")
    print(f"   Year: {result['year']}")
    print(f"   PDF Page: {result['page_number']}")
    print(f"   Rerank Score: {result['rerank_score']:.4f}")
    print(f"   Evidence: {result['text'][:300]}...")

print("\nCross-Encoder reranking test completed!")
