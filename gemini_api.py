"""Gemini helper for country‑specific testing guidance.

This module is used by the Streamlit application to call the official
`google‑genai` SDK. It exposes a single function that returns a tuple
(text, error) so the caller can display guidance or an error message
appropriately.

Authentication:
- Provide your Gemini API key via environment variable as supported by the SDK
  (e.g. ``GOOGLE_API_KEY``) or rely on default credentials.
"""

from typing import Optional, Tuple

try:
    from google import genai
    from google.genai import types
except Exception as e:  # pragma: no cover
    genai = None
    types = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def get_testing_guidance(
    disease: str,
    country: str,
    age_years: int | str,
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, error)."""
    if _IMPORT_ERR is not None:
        return None, f"Gemini SDK not available: {_IMPORT_ERR}"
    if genai is None or types is None:
        return None, "Gemini SDK not available."

    disease = (disease or "").strip()
    country = (country or "").strip()
    age = str(age_years).strip()

    if not disease:
        return None, "Missing disease."
    if not country:
        return None, "Missing country."
    if not age:
        return None, "Missing age."

    client = genai.Client()

    prompt = f"""You are a sexual health assistant.

Task:
Explain the main ways for someone aged {age} in {country} to get tested for {disease}.

Guidelines:
- Start directly with: "In {country}, you can get tested for {disease}..."
- Do NOT use introductory phrases like "I understand" or "Here's a breakdown"
- Address the reader directly using "you"
- Be practical, clear, and specific to {country}
- Mention at least one real public health service, national system, or official clinic type used in {country}
- Explain whether testing may be free, anonymous, or requires a prescription (do not assume it is free unless well established)
- Briefly mention the type of test used (blood test, urine test, swab, rapid test)
- Include common options such as public clinics, hospitals, medical laboratories, sexual health centers, or a general practitioner
- Keep the response short: 2 simple paragraphs. No bullet points.
- No markdown. No bold. No headings.
"""

    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            ),
        )
        text = getattr(resp, "text", None)
        text = (text or "").strip()
        if not text:
            return None, "Empty response from Gemini."
        return text, None
    except Exception as e:
        return None, f"Gemini API call failed: {e}"
