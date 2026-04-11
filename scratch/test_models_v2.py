import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use the EXACT strings from list_models
models_to_try = [
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-3-pro-preview"
]

for m_name in models_to_try:
    print(f"Testing {m_name}...")
    try:
        model = genai.GenerativeModel(m_name)
        response = model.generate_content("Hi")
        print(f"SUCCESS with {m_name}: {response.text[:20]}")
        break
    except Exception as e:
        print(f"FAILED with {m_name}: {e}")
