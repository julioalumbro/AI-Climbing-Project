# check the gemini api integration with a small development request.
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# locate the project root regardless of pycharm's working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load api key from .env
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in .env")

# initialize gemini client
client = genai.Client(api_key=api_key)

# simple text-only test
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Reply with exactly: Gemini API connection successful."
)

print(response.text)