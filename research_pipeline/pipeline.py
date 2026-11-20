
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

from research_pipeline.data_loader import load_research_data
from research_pipeline.retrieval import build_bm25_index


def load_pipeline(api_key):
    """
    Load all research assistant components.
    """

    print("Loading research data...")

    chunks, faiss_index = load_research_data()

    print("Building BM25 index...")

    bm25_index = build_bm25_index(chunks)

    print("Loading Sentence Transformer...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Loading Cross-Encoder...")

    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("Connecting to Gemini...")

    gemini_client = genai.Client(
        api_key=api_key
    )

    print("Research pipeline loaded successfully!")

    return {
        "chunks": chunks,
        "faiss_index": faiss_index,
        "bm25_index": bm25_index,
        "embedding_model": embedding_model,
        "cross_encoder": cross_encoder,
        "gemini_client": gemini_client
    }
