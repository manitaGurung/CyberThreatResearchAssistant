
from research_pipeline.data_loader import load_research_data
from research_pipeline.retrieval import (
    build_bm25_index,
    bm25_retrieval,
    dense_retrieval,
    hybrid_retrieval
)
from sentence_transformers import SentenceTransformer


# Load existing research data
chunks, index = load_research_data()

# Build BM25 index
print("\nBuilding BM25 index...")
bm25 = build_bm25_index(chunks)

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample research question
query = "What machine learning techniques are used for phishing website detection?"

# Test BM25
print("\nTesting BM25...")
bm25_results = bm25_retrieval(query, chunks, bm25, top_k=5)
print("BM25 results:", len(bm25_results))

# Test dense retrieval
print("\nTesting dense retrieval...")
dense_results = dense_retrieval(query, chunks, index, model, top_k=5)
print("Dense results:", len(dense_results))

# Test hybrid retrieval
print("\nTesting hybrid retrieval...")
hybrid_results = hybrid_retrieval(
    query, chunks, index, model, bm25,
    top_k=5, candidate_k=10
)

print("Hybrid results:", len(hybrid_results))

# Display results
print("\nTOP 5 HYBRID RESULTS")

for i, result in enumerate(hybrid_results, start=1):
    print(f"\n{i}. {result['title']}")
    print(f"   Article: {result['article_id']}")
    print(f"   Year: {result['year']}")
    print(f"   Page: {result['page_number']}")
    print(f"   RRF Score: {result['rrf_score']:.4f}")

print("\nRetrieval tests completed!")
