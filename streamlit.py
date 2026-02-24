#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit application for STI/genital pre-triage.

Pipeline:
1) Collect structured patient input.
2) Load candidate diseases directly from sti_structured.csv (36 diseases).
3) Query a language model (MedGemma 27B via Hugging Face pipeline / Gemini) for:
   - A top-3 qualitative differential.
   - The reported uncertainty level.
   - Suggested immediate actions, optionally contextualised by country.

This user interface presents preliminary triage guidance. It does not provide a
definitive diagnosis. Messages about uncertainty and triage are included to
emphasise safety. Optional sensitive fields are handled in a non-intrusive way.
"""

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from gemini_api import get_testing_guidance
except Exception:
    get_testing_guidance = None

import torch
from transformers import pipeline as hf_pipeline

import streamlit as st

try:
    from interpret_image import interpret_images  # type: ignore
except Exception:
    interpret_images = None

import pycountry


# ----------------------------
# Constants
# ----------------------------

COUNTRY_OPTIONS = ["(select)"] + sorted([country.name for country in pycountry.countries])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STI_STRUCTURED = os.path.join(BASE_DIR, "sti_structured.csv")
DEFAULT_GEMINI_RESULTS_CSV = os.path.join(BASE_DIR, "results_gemini.csv")

DEFAULT_MG27B_MODEL = "google/medgemma-27b-text-it"
DEFAULT_MG4B_MODEL = "google/medgemma-1.5-4b-it"

LIKELIHOOD_LABELS = ["very_likely", "likely", "possible", "less_likely", "unlikely"]

LIKELIHOOD_DISPLAY = {
    "very_likely": "Highly similar",
    "likely": "Similar",
    "possible": "Moderately similar",
    "less_likely": "Low similarity",
    "unlikely": "Dissimilar",
}

UNCERTAINTY_DISPLAY = {"low": "Low 🟢", "moderate": "Moderate 🟡", "high": "High 🔴"}
UNCERTAINTY_RENDER = {
    "low": "success",
    "moderate": "warning",
    "high": "error",
}


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class SiteOption:
    code: str
    patient_label: str
    medical_label: str
    retriever_terms: str


SYMPTOMATIC_SITE_OPTIONS: List[SiteOption] = [
    SiteOption("mouth_oral_cavity", "Mouth area", "oral cavity (lips, tongue, gums, palate)",
               "mouth oral cavity lips tongue gingiva palate"),
    SiteOption("throat_pharynx", "Throat", "pharynx / oropharynx",
               "throat pharynx oropharynx tonsillar"),
    SiteOption("eyes_periorbital", "Eyes / around eyes", "ocular/periorbital area",
               "eye eyes ocular conjunctival corneal eyelid periorbital"),
    SiteOption("nose_nasal", "Nose", "nasal area", "nose nasal"),
    SiteOption("ears", "Ears", "auricular/otic area", "ear ears"),
    SiteOption("face", "Face", "facial skin", "face facial"),
    SiteOption("scalp", "Scalp", "scalp", "scalp"),
    SiteOption("neck", "Neck", "cervical/neck skin", "neck"),
    SiteOption("chest", "Chest", "thoracic skin", "chest thoracic"),
    SiteOption("abdomen", "Abdomen", "abdominal skin", "abdomen abdominal"),
    SiteOption("back", "Back", "dorsal skin", "back dorsal"),
    SiteOption("arms_hands", "Arms / hands", "upper limbs including hands/fingers",
               "arm arms forearm hand hands fingers"),
    SiteOption("legs_feet", "Legs / feet", "lower limbs including feet",
               "leg legs foot feet soles palms"),
    SiteOption("penis_glans_shaft", "Penis", "penis (glans/shaft)", "penis glans penile shaft"),
    SiteOption("foreskin_prepuce", "Foreskin", "prepuce / foreskin", "foreskin prepuce"),
    SiteOption("urethral_meatus_urethra", "Urine opening", "urethral meatus / urethra",
               "urethral meatus urethra urethral opening"),
    SiteOption("scrotum_testes", "Scrotum / testicles", "scrotum / testes", "scrotum testicular testes"),
    SiteOption("vulva_labia", "Vulva / labia", "vulva/labia", "vulva vulvar labia"),
    SiteOption("vagina_internal", "Vagina (internal)", "vaginal canal", "vagina vaginal canal"),
    SiteOption("cervix", "Cervix", "cervix", "cervix cervical"),
    SiteOption("anal_perianal", "Anus / around anus", "anal/perianal", "anal perianal"),
    SiteOption("rectal_internal", "Rectum (internal)", "rectal mucosa", "rectal rectum"),
    SiteOption("pubic_groin_perineum", "Pubic/groin/perineal area", "inguinal/perineal area",
               "pubic groin inguinal perineal"),
    SiteOption("buttocks", "Buttocks", "gluteal area", "buttocks gluteal"),
    SiteOption("skin_other", "Other skin area", "other cutaneous site", "skin cutaneous other body site"),
]


# ----------------------------
# Helpers
# ----------------------------

def _site_maps(options: List[SiteOption]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    display: Dict[str, str] = {}
    patient: Dict[str, str] = {}
    medical: Dict[str, str] = {}
    retriever: Dict[str, str] = {}

    for o in options:
        patient[o.code] = o.patient_label
        medical[o.code] = o.medical_label
        retriever[o.code] = o.retriever_terms
        if o.medical_label and o.medical_label.lower() != o.patient_label.lower():
            display[o.code] = f"{o.patient_label} -> {o.medical_label}"
        else:
            display[o.code] = o.patient_label
    return display, patient, medical, retriever


def _extract_json_strict(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if raw is None:
        return None, "Empty response"
    s = raw.strip()
    if not s:
        return None, "Empty response"

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None, "No JSON object found"

    block = m.group(0).strip()
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj, None
        return None, "JSON root is not an object"
    except Exception as e:
        return None, f"JSON parse error: {e}"


def _build_patient_context(
    visual_text: str,
    sensory_text: str,
    symptomatic_sites_codes: List[str],
    age: Optional[int],
    current_country: str,
) -> Tuple[str, Dict[str, Any]]:
    sym_display, sym_patient_map, sym_medical_map, _ = _site_maps(SYMPTOMATIC_SITE_OPTIONS)

    sym_codes = [c for c in symptomatic_sites_codes if c in sym_patient_map]
    sym_patient = [sym_patient_map[c] for c in sym_codes]
    sym_medical = [sym_medical_map[c] for c in sym_codes]
    sym_display_labels = [sym_display[c] for c in sym_codes]

    country_clean = (current_country or "").strip()

    query_parts = [
        f"age: {age}" if age is not None and int(age) > 0 else "",
        f"country: {country_clean or 'unknown'}",
        f"visible symptoms (patient words): {visual_text.strip()}",
        f"sensation symptoms (patient words): {sensory_text.strip()}",
        f"symptom locations (patient terms): {'; '.join(sym_patient)}",
        f"symptom locations (medical terms): {'; '.join(sym_medical)}",
    ]
    query = "\n".join(p for p in query_parts if p and not p.endswith(": "))

    payload = {
        "age": age,
        "current_country": country_clean,
        "visual_text": visual_text.strip(),
        "sensory_text": sensory_text.strip(),
        "symptomatic_sites_codes": sym_codes,
        "symptomatic_sites_display": sym_display_labels,
        "symptomatic_sites_patient_terms": sym_patient,
        "symptomatic_sites_medical_terms": sym_medical,
    }
    return query, payload


def _make_system_prompt() -> str:
    return (
        "You are a medical differential-triage assistant for STI/genital complaints (NOT a diagnostic tool). "
        "Use a neutral, non-judgmental, trauma-informed tone. "
        "Never provide a definitive diagnosis. "
        "Return exactly one JSON object and nothing else. "
        "Use only disease names provided in Options. "
        "Output must be in English only. "
        "No markdown, no code fences, no extra text."
    )


def _make_user_prompt(patient_payload: Dict[str, Any], options: List[str], top_k: int = 3) -> str:
    schema = {
        "top_k": [
            {"disease": "string EXACT from Options",
             "likelihood": "very_likely|likely|possible|less_likely|unlikely",
             "rationale": "one short sentence"}
        ],
        "uncertainty_level": "low|moderate|high",
        "immediate_actions": ["string"],
        "country_specific_actions": ["string"],
        "data_gaps": ["string"],
        "safety_note": "string",
    }

    k_expected = min(max(1, int(top_k)), len(options))

    return (
        "TASK:\n"
        "Produce a provisional differential and triage guidance from a restricted option list.\n\n"
        "STRICT RULES:\n"
        "- Do NOT provide a definitive diagnosis.\n"
        f"- top_k must contain exactly {k_expected} items.\n"
        "- For each item, disease must be copied exactly from Options.\n"
        "- likelihood must be one of: very_likely, likely, possible, less_likely, unlikely.\n"
        "- No numeric probabilities.\n"
        "- Keep rationale short and factual.\n"
        "- Output text must be English only.\n"
        "- uncertainty_level is mandatory.\n"
        "- Respect patient-to-medical term mapping in PATIENT DATA for interpretation.\n"
        "- Treat symptom location selections as optional structured hints; if empty, rely on free-text symptoms.\n"
        "- Use current_country to tailor actions (e.g., where to seek care), not as sole diagnostic evidence.\n"
        "- If country-specific procedures are uncertain, say so and suggest official health resources.\n"
        "- If evidence is limited, reflect this in uncertainty_level and data_gaps.\n\n"
        "- If no visible and sensation symptom, the patient is highly probably healthy.\n\n"
        "PATIENT DATA (structured):\n"
        + json.dumps(patient_payload, ensure_ascii=False)
        + "\n\nOPTIONS:\n"
        + "\n".join(f"- {o}" for o in options)
        + "\n\nOUTPUT JSON SCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


@st.cache_resource(show_spinner=False)
def _load_mg27b_pipeline(hf_token: str) -> Any:
    """Load MedGemma 27B text pipeline once and cache it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_idx = 0 if device == "cuda" else -1
    return hf_pipeline(
        "text-generation",
        model=DEFAULT_MG27B_MODEL,
        torch_dtype=dtype,
        device=device_idx,
        token=hf_token,
    )


def _run_llm(
    hf_token: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Optional[str]]:
    """Call MedGemma 27B via Hugging Face pipeline (text-generation)."""
    if not hf_token.strip():
        return "", "MedGemma 27B API key (Hugging Face token) is empty"

    try:
        pipe = _load_mg27b_pipeline(hf_token.strip())
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = pipe(
            msgs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature) if float(temperature) > 0 else None,
            do_sample=float(temperature) > 0,
        )
        raw = None
        if result and isinstance(result, list) and "generated_text" in result[0]:
            messages = result[0]["generated_text"]
            if messages and isinstance(messages, list):
                raw = messages[-1].get("content", "")
        if not raw or not isinstance(raw, str):
            return "", "Model returned no text content"
        return raw.strip(), None
    except Exception as e:
        return "", str(e)


def _normalize_topk(obj: Dict[str, Any], allowed: List[str], top_k: int) -> List[Dict[str, str]]:
    allowed_set = set(allowed)
    out: List[Dict[str, str]] = []
    seen = set()

    items = obj.get("top_k", [])
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        d = str(it.get("disease", "")).strip()
        if not d or d not in allowed_set or d in seen:
            continue
        label = str(it.get("likelihood", "possible")).strip().lower()
        if label not in LIKELIHOOD_LABELS:
            label = "possible"
        rationale = str(it.get("rationale", "")).strip()
        out.append({"disease": d, "likelihood": label, "rationale": rationale})
        seen.add(d)
        if len(out) >= top_k:
            break
    return out


def _ensure_list_str(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for x in value:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


@st.cache_data(show_spinner=False)
def _load_diseases_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, sep=";")
    return df["disease_full_name"].dropna().tolist()


GEMINI_CSV_COLUMNS = ["disease", "country", "age_group", "guidance_text", "generated_at"]


def _age_group(age: Optional[int]) -> str:
    if age is None:
        return "unknown"
    return "pediatric" if age < 18 else "adult"


def _gemini_csv_key(disease: str, country: str) -> Tuple[str, str]:
    return (disease.strip().lower(), country.strip().lower())


def _load_gemini_csv(csv_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        return cache
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in df.iterrows():
            k = _gemini_csv_key(row.get("disease", ""), row.get("country", ""))
            cache[k] = {
                "disease": row.get("disease", ""),
                "country": row.get("country", ""),
                "age_group": row.get("age_group", ""),
                "guidance_text": row.get("guidance_text", ""),
                "generated_at": row.get("generated_at", ""),
            }
    except Exception:
        pass
    return cache


def _save_gemini_entry(
    csv_path: str,
    disease: str,
    country: str,
    age: Optional[int],
    guidance_text: str,
) -> None:
    row = {
        "disease": disease.strip(),
        "country": country.strip(),
        "age_group": _age_group(age),
        "guidance_text": guidance_text,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GEMINI_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass



# ----------------------------
# Session defaults
# ----------------------------

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "gemini_guidance_cache" not in st.session_state:
    st.session_state["gemini_guidance_cache"] = {}

if "gemini_csv_loaded" not in st.session_state:
    st.session_state["gemini_csv_loaded"] = False

# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="STI / Genital Pre-Triage", page_icon="🩺", layout="wide")
st.title("STI / Genital Pre-Triage")

st.info(
    "This application provides preliminary triage guidance. It does not offer a confirmed diagnosis.",
    icon="ℹ️",
)

st.warning(
    "Seek urgent care now if you have severe pain, high fever, heavy bleeding, "
    "fainting/confusion, or rapidly worsening symptoms.",
    icon="⚠️",
)

with st.sidebar:
    st.subheader("Configuration")

    medgemma_27b_token = st.text_input(
        "MedGemma-27B Access Token [Huggingface](https://huggingface.co/google/medgemma-27b-text-it)",
        value="",
        type="password",
        help="Hugging Face token (Read) for MedGemma-27B text model (differential triage).",
    )

    medgemma_15_token = st.text_input(
        "MedGemma-1.5-4B Access Token [Huggingface](https://huggingface.co/google/medgemma-1.5-4b-it)",
        value="",
        type="password",
        help="Hugging Face token (Read) for MedGemma-1.5-4B image interpretation.",
    )

    gemini_flash_key = st.text_input(
        "Gemini Flash API Key [Gemini](https://aistudio.google.com/api-keys)",
        value="",
        type="password",
        help="Google Gemini Flash 2.5 token."
    )
    if gemini_flash_key:
        os.environ["GOOGLE_API_KEY"] = gemini_flash_key
        os.environ["GEMINI_API_KEY"] = gemini_flash_key

    temperature = 0.0
    max_tokens = 4096


sym_display_map = {o.code: o.patient_label for o in SYMPTOMATIC_SITE_OPTIONS}

st.markdown("### General information")
g1, g2 = st.columns(2)
with g1:
    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=100,
        value=24,
        step=1,
        help="Optional but recommended.",
    )
    age = int(age) if age else None
with g2:
    current_country = st.selectbox(
        "Country",
        options=COUNTRY_OPTIONS,
        index=0,
        help="Optional but recommended. Start typing to search.",
    )
    current_country = "" if current_country == "(select)" else current_country

st.markdown("### Symptoms")

symptomatic_sites = st.multiselect(
    "Organs/regions",
    options=[o.code for o in SYMPTOMATIC_SITE_OPTIONS],
    format_func=lambda c: sym_display_map.get(c, c),
    default=st.session_state.get("global_photo_sites", []),
    key="global_photo_sites",
)

st.caption("Use your own words. You do not need medical vocabulary. For each symptom, include where it occurs.")

c1, c2 = st.columns(2)
with c1:
    visual_text = st.text_area(
        "Visible symptoms (what you can see) *",
        height=220,
        placeholder=(
            "Describe what you can see and where it is.\n"
            "Basic: \"I see a red bump on the outer genital area.\"\n"
            "More detailed: \"I notice small grouped blisters around the anus.\"\n"
        ),
    )
with c2:
    sensory_text = st.text_area(
        "Sensation symptoms (what you can feel) *",
        height=220,
        placeholder=(
            "Describe what you feel and where you feel it.\n"
            "Basic: \"It burns when I pee.\"\n"
            "More detailed: \"I feel itching in the groin and pain in the throat.\"\n"
        ),
    )

# ----------------------------
# Photos
# ----------------------------

st.markdown("### Lesion photos (optional)")
st.caption(
    "Privacy-first: photos are optional. If you choose to add one, it will only be processed when you click Run analysis. "
    "For each added photo, selecting the organ/region(s) shown is required. "
    "Photos are processed locally unless you explicitly submit them."
)

MAX_PHOTO_MB = 10
MAX_PHOTO_BYTES = MAX_PHOTO_MB * 1024 * 1024

if "photo_slots" not in st.session_state:
    st.session_state["photo_slots"] = [{"file": None, "sites": []}]

def _add_photo_slot():
    st.session_state["photo_slots"].append({"file": None, "sites": []})

def _can_add_photo_slot() -> bool:
    i = len(st.session_state["photo_slots"]) - 1
    return st.session_state.get(f"photo_file_{i}") is not None

st.button(
    "➕ Add another photo",
    on_click=_add_photo_slot,
    disabled=not _can_add_photo_slot(),
    help="Upload a photo first to add another one.",
)

for i, slot in enumerate(st.session_state["photo_slots"]):
    st.markdown(f"**Photo {i+1}**")
    up = st.file_uploader(
        f"Upload photo {i+1}",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"photo_file_{i}",
        help=f"Max {MAX_PHOTO_MB}MB per photo.",
    )
    slot["file"] = up

    if up is not None:
        size_bytes = int(getattr(up, "size", 0) or 0)
        if size_bytes > MAX_PHOTO_BYTES:
            st.error(
                f"Photo {i+1} is too large ({size_bytes/1024/1024:.1f}MB). "
                f"Please upload a file ≤ {MAX_PHOTO_MB}MB."
            )
        else:
            show_preview = st.toggle(
                f"Show preview for photo {i+1}",
                value=False,
                key=f"photo_show_{i}",
                help="Off by default for privacy. Turn on to display a small preview.",
            )
            if show_preview:
                st.image(up, width=280)

        slot["sites"] = symptomatic_sites

    st.divider()

st.caption("You can submit as soon as required fields (*) are completed.")
submitted = st.button("🔎 Run analysis (top-3)", type="primary")


# ----------------------------
# Run analysis
# ----------------------------

if submitted:
    if not (visual_text.strip() or sensory_text.strip()):
        st.error("Please provide at least one symptom in either Visible symptoms or Sensation symptoms.")
        st.stop()

    photo_slots = st.session_state.get("photo_slots", [])
    photos_payload = []

    if photo_slots:
        for j, slot in enumerate(photo_slots, start=1):
            f = slot.get("file")
            if f is None:
                continue
            size_bytes = int(getattr(f, "size", 0) or 0)
            if size_bytes > (10 * 1024 * 1024):
                st.error(f"Photo {j} is too large ({size_bytes/1024/1024:.1f}MB). Please upload a file ≤ 10MB.")
                st.stop()

            sites = slot.get("sites") or []
            if len(sites) == 0:
                fname = getattr(f, "name", f"photo_{j}")
                st.error(f"Please select at least one organ/region for Photo {j} ({fname}).")
                st.stop()

            photos_payload.append(
                {
                    "name": getattr(f, "name", f"photo_{j}"),
                    "mime_type": getattr(f, "type", None),
                    "size_bytes": size_bytes,
                    "selected_sites": sites,
                    "bytes": f.getvalue(),
                }
            )

    st.session_state["uploaded_photos"] = photos_payload

    if not os.path.exists(DEFAULT_STI_STRUCTURED):
        st.error(f"File not found: {DEFAULT_STI_STRUCTURED}")
        st.stop()

    image_interpretations: List[Dict[str, Any]] = []
    additional_photo_text = ""

    if photos_payload and medgemma_15_token and medgemma_15_token.strip():
        if interpret_images is None:
            st.warning("Image interpretation is unavailable in this environment.")
        else:
            with st.spinner("Analyzing photos (MedGemma 1.5)…"):
                try:
                    image_interpretations = interpret_images(
                        photos_payload=photos_payload,
                        site_label_map=sym_display_map,
                        sti_csv_path=DEFAULT_STI_STRUCTURED,
                        hf_token=medgemma_15_token.strip(),
                        model_id=DEFAULT_MG4B_MODEL,
                        device=None,
                        max_tokens=512,
                    )
                except Exception as e:
                    st.error(f"Photo interpretation failed: {e}")
                    image_interpretations = []

        if image_interpretations:
            summary_lines: List[str] = []
            for res in image_interpretations:
                if not isinstance(res, dict):
                    continue
                if res.get("error"):
                    summary_lines.append(f"{res.get('photo_name', 'photo')}: Error interpreting image – {res['error']}")
                    continue
                loc = res.get("location") or ", ".join([sym_display_map.get(c, c) for c in res.get("selected_sites", [])])
                desc = (res.get("description", "") or "").strip()
                vis_syms = res.get("visual_symptoms", []) or []
                vis_str = ", ".join([str(s).strip() for s in vis_syms if str(s).strip()])
                if vis_str:
                    summary_lines.append(f"{loc}: {desc}. Descriptors: {vis_str}.")
                else:
                    summary_lines.append(f"{loc}: {desc}.")
            if summary_lines:
                additional_photo_text = "\n\nAdditional observations from uploaded photos:\n" + "\n".join(summary_lines)

    visual_text_with_photos = (visual_text.strip() + additional_photo_text).strip()

    symptomatic_sites_final = symptomatic_sites

    if not symptomatic_sites_final:
        st.info("No optional symptom locations selected. Analysis will rely on your free-text symptom descriptions.")

    candidates = _load_diseases_from_csv(DEFAULT_STI_STRUCTURED)
    if len(candidates) < 3:
        st.error("sti_structured.csv contains fewer than 3 diseases. Please check the file at the project root.")
        st.stop()

    _, patient_payload = _build_patient_context(
        visual_text=visual_text_with_photos,
        sensory_text=sensory_text,
        symptomatic_sites_codes=symptomatic_sites_final,
        age=age,
        current_country=current_country,
    )

    top_k = 3
    user_prompt = _make_user_prompt(patient_payload=patient_payload, options=candidates, top_k=top_k)
    system_prompt = _make_system_prompt()

    with st.spinner("Calling MedGemma 27B (Hugging Face)…"):
        raw, call_err = _run_llm(
            hf_token=medgemma_27b_token,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

    if call_err:
        st.error(f"LLM call failed: {call_err}")
        st.stop()

    parsed, parse_err = _extract_json_strict(raw)
    if parse_err or parsed is None:
        st.error(f"Model response is not usable: {parse_err}")
        with st.expander("Raw model response"):
            st.code(raw or "<empty>")
        st.stop()

    top3 = _normalize_topk(parsed, allowed=candidates, top_k=top_k)
    if len(top3) < top_k:
        st.warning(
            "Top-3 is partially incomplete or contains out-of-options names. "
            "Displayed result has been sanitized by the app."
        )

    uncertainty = str(parsed.get("uncertainty_level", "")).strip().lower()
    safety_note = str(parsed.get("safety_note", "")).strip()
    actions = _ensure_list_str(parsed.get("immediate_actions", []))
    data_gaps = _ensure_list_str(parsed.get("data_gaps", []))

    st.session_state["analysis_done"] = True
    st.session_state["top3"] = top3[:top_k]
    st.session_state["uncertainty"] = uncertainty
    st.session_state["actions"] = actions
    st.session_state["safety_note"] = safety_note
    st.session_state["data_gaps"] = data_gaps

    st.session_state["age_years"] = age
    st.session_state["country_for_gemini"] = (current_country or "").strip()

    st.session_state["patient_payload"] = patient_payload
    st.session_state["user_prompt"] = user_prompt
    st.session_state["parsed"] = parsed
    st.session_state["image_interpretations"] = image_interpretations

    st.success("Analysis completed (provisional differential).")


# ----------------------------
# Display persisted analysis
# ----------------------------

if st.session_state.get("analysis_done"):
    top3 = st.session_state.get("top3", [])
    uncertainty = st.session_state.get("uncertainty", "")
    actions = st.session_state.get("actions", [])
    safety_note = st.session_state.get("safety_note", "")

    unc_label = UNCERTAINTY_DISPLAY.get(uncertainty, uncertainty or "Not provided")
    unc_render = UNCERTAINTY_RENDER.get(uncertainty)
    st.markdown("### Uncertainty")
    if unc_render == "success":
        st.success(unc_label)
    elif unc_render == "warning":
        st.warning(unc_label)
    elif unc_render == "error":
        st.error(unc_label)
    else:
        st.write(unc_label)

    st.markdown("### Top-3 likely conditions")
    rows = []
    for i, item in enumerate(top3, start=1):
        rows.append(
            {
                "Rank": i,
                "Condition": item.get("disease", ""),
                "Likelihood": LIKELIHOOD_DISPLAY.get(item.get("likelihood", ""), item.get("likelihood", "")),
                "Short rationale": item.get("rationale", ""),
            }
        )
    st.data_editor(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        disabled=True)

    top3_diseases = [it["disease"] for it in top3 if it.get("disease")]
    age_years = st.session_state.get("age_years")
    country_for_gemini = st.session_state.get("country_for_gemini", "")

    if get_testing_guidance is None:
        st.info("Gemini guidance is not available in this environment.")
    elif not top3_diseases:
        st.info("No top-3 disease available to generate guidance.")
    elif not country_for_gemini or age_years is None or str(age_years).strip() == "":
        st.info("Add your Country (and Age) to get country-specific testing guidance.")
    else:
        st.markdown("### Testing guidance (Gemini)")

        if not st.session_state["gemini_csv_loaded"]:
            csv_cache = _load_gemini_csv(DEFAULT_GEMINI_RESULTS_CSV)
            for (d, c), entry in csv_cache.items():
                mem_key = f"{entry['disease']}||{entry['country']}"
                if mem_key not in st.session_state["gemini_guidance_cache"]:
                    st.session_state["gemini_guidance_cache"][mem_key] = {
                        "text": entry["guidance_text"],
                        "err": None,
                    }
            st.session_state["gemini_csv_loaded"] = True

        selected_disease = st.selectbox(
            "Choose the condition for testing guidance",
            options=top3_diseases,
            index=0,
            key="selected_disease_for_guidance",
        )

        mem_key = f"{selected_disease}||{country_for_gemini}"

        if mem_key not in st.session_state["gemini_guidance_cache"]:
            with st.spinner(f"Generating testing guidance for '{selected_disease}' (Gemini)..."):
                txt, err = get_testing_guidance(
                    disease=selected_disease,
                    country=country_for_gemini,
                    age_years=age_years,
                )
            st.session_state["gemini_guidance_cache"][mem_key] = {"text": txt or "", "err": err}

            if not err and txt:
                _save_gemini_entry(
                    csv_path=DEFAULT_GEMINI_RESULTS_CSV,
                    disease=selected_disease,
                    country=country_for_gemini,
                    age=age_years,
                    guidance_text=txt,
                )

        res = st.session_state["gemini_guidance_cache"][mem_key]
        if res.get("err"):
            if "RESOURCE_EXHAUSTED" in res["err"]:
                st.warning("Gemini API quota exceeded (free tier limit reached).")
            else:
                st.warning(f"Gemini guidance unavailable for '{selected_disease}': {res['err']}")
        elif res.get("text"):
            st.markdown(res["text"])
        else:
            st.info(f"No guidance text returned for '{selected_disease}'.")

    if actions:
        st.markdown("### Immediate suggested actions")
        for act in actions:
            st.write(f"- {act}")

    if safety_note:
        st.markdown("### Safety note")
        st.write(safety_note)

    debug = False
    if debug:
        with st.expander("Technical details (debug)"):
            st.markdown("**Structured patient payload sent to model**")
            st.json(st.session_state.get("patient_payload", {}))

            st.markdown("**User prompt sent to model (preview)**")
            st.code(st.session_state.get("user_prompt", ""))

            st.markdown("**Model JSON response**")
            st.json(st.session_state.get("parsed", {}))

            try:
                image_interpretations = st.session_state.get("image_interpretations", [])
                if image_interpretations:
                    st.markdown("**Image interpretations (MedGemma-1.5)**")
                    st.json(image_interpretations)
            except Exception:
                pass
