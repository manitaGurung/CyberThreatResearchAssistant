from pathlib import Path
import pickle
import faiss


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Artifact paths
DATA_DIR = PROJECT_ROOT / "data" / "processed_data"

CHUNKS_PATH = DATA_DIR / "sentence_based_chunks.pkl"
FAISS_PATH = DATA_DIR / "sentence_based_faiss.index"


def load_research_data():
    """Load the sentence-based chunks and FAISS index."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")

    # Load sentence-based chunks
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    # Load FAISS index
    index = faiss.read_index(str(FAISS_PATH))

    # Verify consistency
    if len(chunks) != index.ntotal:
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but "
            f"{index.ntotal} FAISS vectors."
        )

    print(f"Loaded {len(chunks):,} chunks.")
    print(f"Loaded {index.ntotal:,} FAISS vectors.")

    return chunks, index