
import tomllib
from pathlib import Path

from research_pipeline.pipeline import load_pipeline

# Load Gemini API key
secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"

with open(secrets_path, "rb") as file:
    secrets = tomllib.load(file)

api_key = secrets["GEMINI_API_KEY"]

# Load all pipeline components
pipeline = load_pipeline(api_key)

# Verify components
print("\nPIPELINE COMPONENT CHECK")

for component, value in pipeline.items():
    print(f"{component}: {'Loaded successfully' if value is not None else 'Failed'}")

print("\nTotal research chunks:", len(pipeline["chunks"]))
print("Total FAISS vectors:", pipeline["faiss_index"].ntotal)

print("\nComplete pipeline test successful!")
