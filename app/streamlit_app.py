

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import tomllib

from research_pipeline.pipeline import load_pipeline
from research_pipeline.answer_generator import generate_evidence_grounded_answer


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cyber Threat Research Assistant",
    page_icon="🔐",
    layout="wide"
)


# --------------------------------------------------
# 2. Application title and introduction
# --------------------------------------------------

st.title("🔐 Cyber Threat Research Assistant")

st.markdown(
    """
    ### Evidence-Grounded Academic Research Assistant

    Explore cybersecurity research through hybrid information retrieval,
    Cross-Encoder reranking, and large language model-based answer generation.

    Ask a research question and receive an answer supported by passages
    retrieved from the research article collection.
    """
)

st.divider()


# --------------------------------------------------
# 3. Load Gemini API key
# --------------------------------------------------

SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"


def get_api_key():
    with open(SECRETS_PATH, "rb") as file:
        secrets = tomllib.load(file)

    return secrets["GEMINI_API_KEY"]


# --------------------------------------------------
# 4. Load research pipeline only once
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_pipeline(api_key):
    return load_pipeline(api_key)


try:
    with st.spinner("Loading research models and database..."):
        pipeline = initialize_pipeline(get_api_key())

    st.success("Research assistant is ready!")

except Exception as e:
    st.error(f"Unable to initialize the research assistant: {e}")
    st.stop()


# --------------------------------------------------
# 5. Research question input
# --------------------------------------------------

st.subheader("Ask a Research Question")

query = st.text_area(
    "Enter your cybersecurity research question:",
    placeholder="Example: What machine learning techniques are commonly used for phishing website detection?",
    height=120
)

generate_button = st.button(
    "Generate Research Answer",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# 6. Generate evidence-grounded answer
# --------------------------------------------------

if generate_button:

    if not query.strip():
        st.warning("Please enter a research question first.")

    else:

        try:
            with st.spinner(
                "Searching research articles, reranking evidence, and generating your answer..."
            ):

                result = generate_evidence_grounded_answer(
                    query=query.strip(),
                    chunks=pipeline["chunks"],
                    faiss_index=pipeline["faiss_index"],
                    embedding_model=pipeline["embedding_model"],
                    bm25_index=pipeline["bm25_index"],
                    cross_encoder=pipeline["cross_encoder"],
                    gemini_client=pipeline["gemini_client"]
                )

            st.divider()

            # Display generated answer
            st.subheader("Generated Research Answer")

            if result["answer"].startswith("Gemini API request failed:"):
                st.error(result["answer"])
            else:
                st.markdown(result["answer"])

            # Display retrieved evidence
            st.divider()
            st.subheader("Retrieved Research Evidence")

            st.caption(
                f"{len(result['evidence'])} passages retrieved and reranked."
            )

            for i, passage in enumerate(result["evidence"], start=1):

                year = passage["year"]

                if year is not None:
                    try:
                        if str(year).lower() != "nan":
                            year = int(float(year))
                        else:
                            year = "Not available"
                    except (ValueError, TypeError):
                        year = "Not available"
                else:
                    year = "Not available"

                with st.expander(
                    f"Evidence {i}: {passage['title']}"
                ):

                    st.write("**Article ID:**", passage["article_id"])
                    st.write("**Publication Year:**", year)
                    st.write("**PDF Page:**", passage["page_number"])
                    st.write(
                        "**Cross-Encoder Score:**",
                        round(float(passage["rerank_score"]), 4)
                    )

                    st.markdown("**Supporting Passage:**")
                    st.write(passage["text"])

        except Exception as e:
            st.error(f"Answer generation failed: {e}")


# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Cyber Threat Research Assistant | "
    "Hybrid Retrieval + Cross-Encoder Reranking + Gemini"
)
