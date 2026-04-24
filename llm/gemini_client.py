"""
llm/gemini_client.py — Google Gemini API wrapper

Provides both sync and async interfaces for text generation.
"""

import logging
import sys
import base64
import io
from pathlib import Path

import google.generativeai as genai
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEMINI_API_KEY, GEMINI_MODEL
from llm.prompts import SYSTEM_PROMPT

log = logging.getLogger(__name__)

# Configure SDK once at import
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    log.warning("GEMINI_API_KEY not set — LLM calls will fail.")

# Legal context often triggers false safety blocks (e.g., discussing "harassment" or "injury")
# We set all categories to BLOCK_NONE to ensure legal analysis is never interrupted.
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

_model_instance = None


def _get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            safety_settings=SAFETY_SETTINGS,
        )
    return _model_instance


def generate_text_sync(prompt: str, max_tokens: int = 2048, attached_image_b64: str | None = None) -> str:
    """Synchronous text generation via Gemini."""
    try:
        model = _get_model()
        contents = [prompt]
        
        if attached_image_b64:
            if "," in attached_image_b64:
                attached_image_b64 = attached_image_b64.split(",")[1]
            img_bytes = base64.b64decode(attached_image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            contents = [prompt, img]
            
        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,   # low temperature for legal accuracy
            ),
        )
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            log.info(f"    [GEMINI USAGE] Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count} | Total: {usage.total_token_count}")
        
        try:
            return response.text
        except ValueError:
            # If the response was blocked
            if response.candidates:
                reason = response.candidates[0].finish_reason
                log.warning(f"Gemini response blocked. Reason: {reason}")
                return "I apologize, but I cannot provide an answer to this specific query due to safety filtering. Please try rephrasing."
            return "Unexpected empty response from model."
    except Exception as e:
        log.error(f"Gemini generation error: {e}")
        raise


async def generate_text(prompt: str, max_tokens: int = 2048, attached_image_b64: str | None = None) -> str:
    """Async text generation via Gemini (runs sync in thread pool)."""
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: generate_text_sync(prompt, max_tokens, attached_image_b64)
    )
    
def extract_image_text_sync(image_path: Path) -> str:
    """Extract raw text from an image utilizing Gemini Vision locally."""
    try:
        model = _get_model()
        img = Image.open(str(image_path))
        response = model.generate_content(
            ["Please accurately transcribe all text visible in this image. Do not include external commentary. Ensure every sentence is captured.", img],
             generation_config=genai.GenerationConfig(temperature=0.0)
        )
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            log.info(f"    [GEMINI OCR USAGE] Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count} | Total: {usage.total_token_count}")
        return response.text
    except Exception as e:
        log.error(f"OCR Generation Error: {e}")
        return ""


def generate_with_history(messages: list[dict], max_tokens: int = 2048) -> str:
    """
    Generate with conversation history.
    messages: list of {"role": "user"|"model", "parts": [str]}
    """
    try:
        model = _get_model()
        chat = model.start_chat(history=messages[:-1])
        last_msg = messages[-1]["parts"][0]
        response = chat.send_message(
            last_msg,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
            ),
        )
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            log.info(f"    [GEMINI CHAT USAGE] Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count} | Total: {usage.total_token_count}")
        
        try:
            return response.text
        except ValueError:
            if response.candidates:
                log.warning(f"Gemini chat response blocked. Reason: {response.candidates[0].finish_reason}")
                return "I apologize, but I cannot continue this part of the conversation due to safety filtering. Let's try another topic."
            return "Unexpected empty response from model."
    except Exception as e:
        log.error(f"Gemini chat error: {e}")
        raise
