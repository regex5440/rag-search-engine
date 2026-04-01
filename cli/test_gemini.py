import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

responseObj = client.models.generate_content(model="gemma-3-27b-it", contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")

print(responseObj.text)
metadata = responseObj.usage_metadata
if metadata is not None:
    print(f"\nPrompt tokens: {metadata.prompt_token_count}\nResponse tokens: {metadata.candidates_token_count}")
