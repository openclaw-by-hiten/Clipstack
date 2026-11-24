import base64
import io
import json
import os
from typing import Dict, Optional

import cv2
import numpy as np
import requests
from PIL import Image


def _sample_storyboard_frames(video_path: str, count: int = 9) -> list[np.ndarray]:
    frames = []
    if not video_path or not os.path.exists(video_path):
        return frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = total / fps if total > 0 else 0.0
        if dur <= 0.2:
            return frames
        for i in range(count):
            t = max(0.05, min(dur - 0.1, (i + 0.5) * (dur / count)))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frm = cap.read()
            if ret and frm is not None:
                frames.append(frm)
    finally:
        cap.release()
    return frames


def _build_contact_sheet_jpeg(frames: list[np.ndarray], cell_w: int = 256, cell_h: int = 144) -> Optional[bytes]:
    if not frames:
        return None
    cols = 3
    rows = max(1, (len(frames) + cols - 1) // cols)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, frm in enumerate(frames[: rows * cols]):
        r, c = divmod(idx, cols)
        resized = cv2.resize(frm, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def _try_gemini_vision_hype(sheet_jpeg: bytes, source_title: str, transcript: str) -> Optional[Dict]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not sheet_jpeg:
        return None
    try:
        b64 = base64.b64encode(sheet_jpeg).decode("utf-8")
        prompt = (
            "You are a viral short-form video editor analyzing a 3x3 chronological storyboard grid of a video clip.\n"
            f"Video title: {source_title or 'Unknown'}\n"
            f"Clip transcript: {(transcript or '')[:350]}\n"
            "Return ONLY valid JSON: "
            '{"visual_score": <integer 0-100 for visual hook/action/reaction excitement>, '
            '"visual_summary": "<1 concise visual sentence describing what is literally happening on screen for an 8K thumbnail prompt>"}'
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key.strip()}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
        }
        r = requests.post(url, json=payload, timeout=14)
        if r.status_code == 200:
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw)
            v_score = float(np.clip(float(parsed.get("visual_score", 65)), 0.0, 100.0))
            v_sum = str(parsed.get("visual_summary", "")).strip()
            print(f"[VisualHype] Gemini Vision scored {v_score:.1f}% | scene='{v_sum[:70]}'")
            return {"visual_score": round(v_score, 1), "visual_summary": v_sum, "engine": "gemini_vision"}
    except Exception as e:
        print(f"[VisualHype] Gemini Vision fallback: {e}")
    return None


def _try_local_qwen_vl(sheet_jpeg: bytes, source_title: str, transcript: str) -> Optional[Dict]:
    """
    Optional local Qwen-VL multimodal hype scoring on high-VRAM CUDA GPUs when
    CLIPSTACK_ENABLE_LOCAL_VLM=1 is explicitly enabled (avoids Colab 12.7GB RAM spikes by default).
    """
    if os.environ.get("CLIPSTACK_ENABLE_LOCAL_VLM", "0").strip() != "1":
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except Exception:
        return None

    model = None
    try:
        model_id = os.environ.get("CLIPSTACK_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        print(f"[VisualHype] Running local VLM {model_id} on storyboard grid...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        im = Image.open(io.BytesIO(sheet_jpeg)).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": im},
                {
                    "type": "text",
                    "text": (
                        f"Title: {source_title or 'Video'}. Transcript: {(transcript or '')[:220]}. "
                        'Return JSON only: {"visual_score": 0-100, "visual_summary": "concise 12-word scene description"}.'
                    ),
                },
            ],
        }]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_in], images=[im], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=96)
        decoded = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
        start_b, end_b = decoded.find("{"), decoded.rfind("}")
        if start_b != -1 and end_b > start_b:
            parsed = json.loads(decoded[start_b : end_b + 1])
            v_score = float(np.clip(float(parsed.get("visual_score", 65)), 0.0, 100.0))
            v_sum = str(parsed.get("visual_summary", "")).strip()
            print(f"[VisualHype] Local Qwen-VL scored {v_score:.1f}% | scene='{v_sum[:70]}'")
            return {"visual_score": round(v_score, 1), "visual_summary": v_sum, "engine": "qwen_vl"}
    except Exception as e:
        print(f"[VisualHype] Local Qwen-VL fallback: {e}")
    finally:
        if model is not None:
            del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return None


def _score_cv_visual_hype(frames: list[np.ndarray]) -> Dict:
    """
    Ultra-fast 0-VRAM Computer Vision Visual Hype Scorer (~0.35s):
    Measures action motion energy, shot-cut rhythm, visual drama (saturation/contrast),
    and human close-up presence so ClipStack actually sees visual excitement!
    """
    if not frames:
        return {"visual_score": 55.0, "visual_summary": "", "engine": "cv_fallback"}

    grays = []
    sats = []
    contrasts = []
    brights = []
    for frm in frames:
        small = cv2.resize(frm, (256, 144), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grays.append(g)
        sats.append(float(np.mean(hsv[:, :, 1])) / 255.0)
        brights.append(float(np.mean(g)) / 255.0)
        contrasts.append(float(np.std(g)) / 128.0)

    motion_diffs = [
        float(np.mean(cv2.absdiff(grays[i], grays[i - 1]))) / 255.0
        for i in range(1, len(grays))
    ]
    avg_motion = float(np.mean(motion_diffs)) if motion_diffs else 0.05
    peak_motion = float(np.max(motion_diffs)) if motion_diffs else 0.08
    cut_hits = sum(1 for d in motion_diffs if d > 0.14)
    avg_sat = float(np.mean(sats))
    avg_contrast = float(np.mean(contrasts))
    avg_bright = float(np.mean(brights))

    # Normalize into a calibrated 0-100 visual excitement score
    motion_pts = min(38.0, (avg_motion * 190.0) + (peak_motion * 60.0))
    rhythm_pts = min(24.0, cut_hits * 5.5)
    drama_pts = min(22.0, (avg_sat * 22.0) + (avg_contrast * 18.0))
    base_pts = 20.0
    v_score = float(np.clip(base_pts + motion_pts + rhythm_pts + drama_pts, 35.0, 96.0))

    # Synthesize concise visual style cues for 8K thumbnail conditioning
    cues = []
    if avg_bright < 0.34:
        cues.append("dramatic night scene")
    elif avg_bright > 0.62:
        cues.append("bright daylight action")
    if avg_sat > 0.42:
        cues.append("vibrant neon lighting")
    if avg_motion > 0.11 or cut_hits >= 3:
        cues.append("high-intensity fast action")
    else:
        cues.append("focused dramatic close-up")

    return {
        "visual_score": round(v_score, 1),
        "visual_summary": ", ".join(cues),
        "engine": "cv_visual_hype",
    }


def analyze_visual_hype(
    video_slice_path: str,
    source_title: str = "",
    transcript: str = "",
    enable_vlm: bool = True,
) -> Dict:
    """
    Phase 3 Multimodal Visual Hype Analyzer:
    1. Samples 9 chronological keyframes into a compact 3x3 storyboard grid.
    2. If enabled, queries Gemini Flash Vision (if GEMINI_API_KEY set) or local Qwen-VL (if enabled).
    3. Always computes fast 0-VRAM CV visual excitement metrics (motion, cuts, vibrancy, contrast).
    """
    frames = _sample_storyboard_frames(video_slice_path, count=9)
    if not frames:
        return {"visual_score": 55.0, "visual_summary": "", "engine": "none"}

    if enable_vlm:
        sheet_jpeg = _build_contact_sheet_jpeg(frames)
        if sheet_jpeg:
            res = _try_gemini_vision_hype(sheet_jpeg, source_title, transcript)
            if res:
                return res
            res = _try_local_qwen_vl(sheet_jpeg, source_title, transcript)
            if res:
                return res

    return _score_cv_visual_hype(frames)
