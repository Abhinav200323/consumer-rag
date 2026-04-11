import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

print(f"Key found: {'Yes' if key else 'No'}")
if key:
    print(f"Key starts with: {key[:7]}...")
    print(f"Key length: {len(key)}")
print(f"Model: {model}")
