import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash-latest"]

for m_name in models_to_try:
    print(f"Testing {m_name}...")
    try:
        model = genai.GenerativeModel(m_name)
        response = model.generate_content("Hi")
        print(f"SUCCESS with {m_name}: {response.text[:20]}")
        break
    except Exception as e:
        print(f"FAILED with {m_name}: {e}")
