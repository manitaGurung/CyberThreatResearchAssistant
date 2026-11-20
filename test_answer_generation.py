
import tomllib
from pathlib import Path

from research_pipeline.pipeline import load_pipeline
from research_pipeline.answer_generator import generate_evidence_grounded_answer

# Load API key
secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"

with open(secrets_path, "rb") as file:
    secrets = tomllib.load(file)

# Load pipeline
pipeline = load_pipeline(secrets["GEMINI_API_KEY"])

# Research question
query = "What machine learning techniques are commonly used for phishing website detection?"

print("\nGenerating evidence-grounded answer...\n")

# Generate answer
result = generate_evidence_grounded_answer(
    query=query,
    chunks=pipeline["chunks"],
    faiss_index=pipeline["faiss_index"],
    embedding_model=pipeline["embedding_model"],
    bm25_index=pipeline["bm25_index"],
    cross_encoder=pipeline["cross_encoder"],
    gemini_client=pipeline["gemini_client"]
)

# Display answer
print("=" * 80)
print("RESEARCH QUESTION")
print("=" * 80)
print(result["question"])

print("\n" + "=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(result["answer"])

print("\n" + "=" * 80)
print("RETRIEVED AND RERANKED EVIDENCE")
print("=" * 80)

for i, passage in enumerate(result["evidence"], start=1):
    print(f"\nEvidence {i}")
    print("Article:", passage["title"])
    print("Year:", passage["year"])
    print("PDF Page:", passage["page_number"])
    print("Rerank Score:", passage.get("rerank_score"))
    print("Passage:", passage["text"][:500])

print("\nAnswer generation test completed!")
