#!/usr/bin/env python3
"""
interpret_image

This module provides helper routines to analyse user‑supplied lesion photos
using a vision–enabled MedGemma model (4B variant). Given one or more
uploaded photos and the anatomical region(s) selected by the user for each
image, the helper functions will call the OpenAI‑compatible API serving
unsloth/medgemma-1.5-4b-it. The model is prompted to describe the
morphology of the lesion and to select the most plausible visual symptom
descriptors from a controlled vocabulary derived from `sti_structured.csv`.
"""

import ast
import base64
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from PIL import Image  # type: ignore
import torch  # type: ignore
from transformers import pipeline  # type: ignore


def load_visual_symptoms(sti_csv_path: str) -> List[str]:
    if not os.path.exists(sti_csv_path):
        raise FileNotFoundError(f"sti_structured.csv not found at {sti_csv_path}")
    df = pd.read_csv(sti_csv_path)
    symptoms: List[str] = []
    for _, row in df.iterrows():
        raw = row.get("visual_symptoms_by_stage", None)
        if pd.isna(raw) or not isinstance(raw, str) or raw.strip() == "[]":
            continue
        try:
            stages = ast.literal_eval(raw)
        except Exception:
            continue
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            for key in ("visual_genital", "visual_oral", "visual_other"):
                items = stage.get(key)
                if not items:
                    continue
                for s in items:
                    if not isinstance(s, str):
                        continue
                    cleaned = s.strip()
                    if cleaned:
                        symptoms.append(cleaned)
    unique = sorted(set(symptoms))
    return unique


def _extract_json_strict(raw: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
    m = None
    try:
        import re

        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    except Exception:
        m = None
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


def interpret_single_image(
    image_bytes: bytes,
    selected_sites: List[str],
    site_label_map: Dict[str, str],
    symptoms_list: List[str],
    pipe: Any,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Call MedGemma 1.5 4B via Hugging Face pipeline on a single image.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        selected_sites: List of site codes corresponding to anatomical regions.
        site_label_map: Mapping from site code to display label.
        symptoms_list: Controlled vocabulary of visual symptom descriptors.
        pipe: A pre‑initialised Hugging Face pipeline for image-text-to-text.
        max_tokens: Maximum number of new tokens to generate. Defaults to 2048.

    Returns:
        A dictionary with keys location, description, visual_symptoms.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    if not selected_sites:
        raise ValueError("selected_sites is empty")
    if not symptoms_list:
        raise ValueError("symptoms_list is empty")
    location_labels = [site_label_map.get(code, code) for code in selected_sites]
    seen = set()
    location_ordered: List[str] = []
    for label in location_labels:
        if label in seen:
            continue
        seen.add(label)
        location_ordered.append(label)
    location_str = ", ".join(location_ordered)
    try:
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")
    system_prompt = (
        "You are a specialist medical image interpretation assistant. "
        "Your role is to perform a neutral, factual, morphological assessment of skin, mucosal or genital lesions. "
        "Never attempt to make a diagnosis or suggest a disease name. "
        "Focus strictly on describing the visual appearance and selecting relevant descriptors from a provided list. "
        "Only reply with a JSON object matching the requested schema and no other commentary."
    )
    symptoms_concat = "; ".join(symptoms_list)
    user_prompt_text = (
        f"The following image shows a lesion located in: {location_str}. "
        f"Analyse the lesion and: (1) briefly describe its morphology (colour, shape, number, pattern, any surface changes) in one concise sentence; "
        f"(2) From the list of possible visual symptoms provided below, select all descriptors that match what you see. "
        f"Use ONLY descriptors from the list and DO NOT invent new terms. If none of the descriptors apply, return an empty list.\n\n"
        f"Possible visual symptom descriptors:\n{symptoms_concat}\n\n"
        "Return your answer as a JSON object with exactly the keys: "
        "'location' (string, copy of the provided location), "
        "'description' (string, your short morphological description), and "
        "'visual_symptoms' (array of strings containing only descriptors from the list). "
        "Do not include any other keys or text."
    )
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt_text},
            {"type": "image", "image": image},
        ]},
    ]
    try:
        result = pipe(text=msgs, max_new_tokens=int(max_tokens))
    except Exception as exc:
        raise RuntimeError(f"Vision model call failed: {exc}")
    raw = None
    try:
        if result and isinstance(result, list) and "generated_text" in result[0]:
            messages = result[0]["generated_text"]
            if messages and isinstance(messages, list):
                raw = messages[-1]["content"]
    except Exception:
        raw = None
    if not raw or not isinstance(raw, str):
        raise ValueError("Vision model returned no text content")
    raw_text = raw.strip()
    obj, err = _extract_json_strict(raw_text)
    if err or obj is None:
        raise ValueError(f"Vision model did not return valid JSON: {err or 'unknown error'}; raw response: {raw_text[:200]}...")
    loc_val = str(obj.get("location", "")).strip()
    desc_val = str(obj.get("description", "")).strip()
    vis_list = obj.get("visual_symptoms", [])
    if not loc_val:
        loc_val = location_str
    if not isinstance(vis_list, list):
        vis_list = []
    out_vis: List[str] = []
    for item in vis_list:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and s in symptoms_list:
            out_vis.append(s)
    return {
        "location": loc_val,
        "description": desc_val,
        "visual_symptoms": out_vis,
    }


def interpret_images(
    photos_payload: List[Dict[str, Any]],
    site_label_map: Dict[str, str],
    sti_csv_path: str,
    hf_token: str,
    model_id: str = "google/medgemma-1.5-4b-it",
    device: Optional[str] = None,
    max_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """Interpret multiple uploaded photos using a Hugging Face pipeline.

    Args:
        photos_payload: List of dictionaries representing each uploaded photo.
        site_label_map: Mapping from site code to user‑facing label.
        sti_csv_path: Path to sti_structured.csv used to extract visual symptom descriptors.
        hf_token: Hugging Face token used to authenticate access to the MedGemma 1.5 model. 
            If empty or invalid, model loading will fail.
        model_id: Identifier of the vision model to load. 
            Defaults to google/medgemma-1.5-4b-it.
        device: Optional device specifier.
        max_tokens: Maximum number of tokens to generate from the model.
            Defaults to 2048

    Returns:
        A list of dictionaries.
    """
    results: List[Dict[str, Any]] = []
    if not photos_payload:
        return results
    symptoms_list = load_visual_symptoms(sti_csv_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_idx = 0 if device == "cuda" else -1
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=dtype,
            device=device_idx,
            token=hf_token,
        )
    except Exception as e:
        for photo in photos_payload:
            name = photo.get("name") or "unknown_photo"
            selected_sites = photo.get("selected_sites", []) or []
            results.append({
                "photo_name": name,
                "selected_sites": selected_sites,
                "error": f"Failed to load model: {e}",
            })
        return results
    for photo in photos_payload:
        image_bytes = photo.get("bytes")
        selected_sites = photo.get("selected_sites", []) or []
        name = photo.get("name") or "unknown_photo"
        if not image_bytes or not selected_sites:
            continue
        try:
            res = interpret_single_image(
                image_bytes=image_bytes,
                selected_sites=selected_sites,
                site_label_map=site_label_map,
                symptoms_list=symptoms_list,
                pipe=pipe,
                max_tokens=max_tokens,
            )
            res["photo_name"] = name
            res["selected_sites"] = selected_sites
            results.append(res)
        except Exception as e:
            results.append({
                "photo_name": name,
                "selected_sites": selected_sites,
                "error": str(e),
            })
    return results


__all__ = [
    "load_visual_symptoms",
    "interpret_single_image",
    "interpret_images",
]
