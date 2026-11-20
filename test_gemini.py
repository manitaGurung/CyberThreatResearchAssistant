
import tomllib
from pathlib import Path
from google import genai

# Load the API key securely
secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"

with open(secrets_path, "rb") as file:
    secrets = tomllib.load(file)

api_key = secrets["GEMINI_API_KEY"]

# Initialize Gemini client
gemini_client = genai.Client(api_key=api_key)

# Test the original Interactions API
try:
    response = gemini_client.interactions.create(
        model="gemini-3.5-flash-lite",
        input="Reply with exactly: Gemini connection successful."
    )

    print("Gemini API connection successful!")
    print("Response:", response.output_text)

except Exception as e:
    print("Gemini API test failed.")
    print("Error:", str(e))
