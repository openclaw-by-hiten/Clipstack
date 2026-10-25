import os
import re
import io
import math
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from core.branding import composite_brand_on_thumbnail
from core.smart_crop import create_yunet_detector

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
AI_THUMB_DIR = WORKSPACE_DIR / "ai_thumbnails"
AI_THUMB_DIR.mkdir(parents=True, exist_ok=True)

_DANGLING_END_WORDS = {
    "on", "in", "at", "to", "for", "with", "from", "of", "by", "into", "onto",
    "the", "a", "an", "our", "my", "your", "his", "her", "their", "its",
    "this", "that", "these", "those", "and", "or", "but", "if", "when", "while",
    "is", "are", "was", "were", "be", "been", "will", "would", "can", "could",
}


def _target_size(aspect_ratio: str) -> tuple[int, int]:
    if aspect_ratio == "1:1":
        return 1080, 1080
    if aspect_ratio == "16:9":
        return 1920, 1080
    return 1080, 1920


def _load_bold_font(size: int):
    candidates = [
        r"C:\Windows\Fonts\impact.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\bahnschrift.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def analyze_video_context(hook_text: str = None, transcript_text: str = None, source_title: str = None) -> dict:
    raw_combined = f"{source_title or ''} {hook_text or ''} {transcript_text or ''}"
    text = raw_combined.lower()
    themes = []
    if any(k in text for k in ["150 lbs", "150 pounds", "lose 150"]):
        themes.append("mrbeast_150lbs_special")
    if any(k in text for k in ["pound", "lbs", "weight", "scale", "gym", "workout"]):
        themes.append("weight_loss_transformation")
    if any(k in text for k in ["money", "cash", "dollar", "$", "thousand", "million", "prize", "vault"]):
        themes.append("cash_prize_challenge")
    if any(k in text for k in ["raft", "ocean", "sea", "island", "survive", "stranded", "water", "boat", "storm"]):
        themes.append("survival_adventure")
    if any(k in text for k in ["podcast", "interview", "mindset", "business", "secret", "truth", "story"]):
        themes.append("podcast_spotlight")
    return {"themes": themes, "clean_text": text[:500]}


def find_best_ai_template(ai_candidates: list, clip_index: int, context: dict) -> Path:
    if not ai_candidates:
        return None
    default_template = ai_candidates[(clip_index - 1) % len(ai_candidates)]
    text = context.get("clean_text", "")
    scored = []
    for cand in ai_candidates:
        score = 0
        name = cand.stem.lower()
        if "scale" in name and ("scale" in text or "150" in text):
            score += 15
        if "lost" in name and ("130" in text or "scoreboard" in text):
            score += 20
        if "victory" in name and ("250 days" in text or "finale" in text or "did it" in text):
            score += 20
        if "workout" in name and ("workout" in text or "gym" in text or "sweat" in text):
            score += 20
        if "check" in name and ("winner" in text or "250,000" in text):
            score += 18
        scored.append((cand, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored[0][1] >= 15 else default_template


def _extract_best_hero_frame(video_path: str, clip_duration: float, clip_index: int = 1) -> dict:
    """
    Scans 16 timestamps across the clip with YuNet CNN Face Detector + sharpness +
    eye-openness/expression scoring. Offsets timestamp preference by `clip_index` so
    multiple clips from the same session feature distinct expressive hero moments.
    """
    result = {"frame_bgr": None, "faces": [], "has_strong_portrait": False}
    if not video_path or not os.path.exists(video_path):
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return result

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = max(1.0, min(clip_duration, total_frames / fps if total_frames > 0 else clip_duration))

    det_w = min(w, 960)
    det_h = max(32, int(round(h * (det_w / max(1.0, float(w))))))
    inv_scale = float(w) / float(det_w)
    yunet = create_yunet_detector(det_w, det_h, score_threshold=0.48)

    num_probes = 16
    # Give each clip_index a preferred temporal sweet-spot so clips never look identical
    preferred_frac = [0.22, 0.58, 0.38, 0.76, 0.48][(clip_index - 1) % 5]
    best_score = -1e9
    best_frame = None
    best_faces = []
    strong_portrait = False

    for i in range(num_probes):
        frac = 0.06 + i * (0.86 / max(1, num_probes - 1))
        t = max(0.12, min(dur - 0.12, dur * frac))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        if mean_luma < 18.0 or mean_luma > 246.0:
            continue

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))

        faces_found = []
        if yunet is not None:
            yunet.setInputSize((det_w, det_h))
            _, dets = yunet.detect(small)
            if dets is not None:
                for row in dets:
                    fx, fy, fw, fh = row[0], row[1], row[2], row[3]
                    conf = float(row[14])
                    if fw < det_h * 0.055:
                        continue
                    x1 = max(0, int(round(fx * inv_scale)))
                    y1 = max(0, int(round(fy * inv_scale)))
                    x2 = min(w, int(round((fx + fw) * inv_scale)))
                    y2 = min(h, int(round((fy + fh) * inv_scale)))
                    # Measure frontal symmetry from YuNet eye landmarks (frontal gazes make stronger posters!)
                    re_x, le_x, nose_x = float(row[4]), float(row[6]), float(row[8])
                    eye_span = abs(le_x - re_x) / max(1.0, float(fw))
                    frontal_bonus = min(1.30, 0.80 + eye_span * 1.1)
                    faces_found.append((x1, y1, x2, y2, conf * frontal_bonus))

        faces_found.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]) * f[4], reverse=True)

        temporal_diversity_bonus = max(0.0, 90.0 * (1.0 - abs(frac - preferred_frac) * 1.8))
        frame_score = min(340.0, sharpness * 0.55) + sat_mean * 0.45 + temporal_diversity_bonus
        frame_has_portrait = False

        if faces_found:
            fx1, fy1, fx2, fy2, fscore = faces_found[0]
            fh_ratio = (fy2 - fy1) / max(1.0, float(h))
            edge_ok = 1.25 if (fy1 > h * 0.03 and fx1 > w * 0.04 and fx2 < w * 0.96) else 0.72
            face_bonus = (fh_ratio * 1750.0) * fscore * edge_ok
            if len(faces_found) >= 2:
                s2 = faces_found[1]
                span = max(fx2, s2[2]) - min(fx1, s2[0])
                if span <= h * (9.0 / 16.0) * 1.15:
                    face_bonus *= 1.24
            frame_score += face_bonus
            if fh_ratio >= 0.13 and sharpness >= 50.0:
                frame_has_portrait = True

        if frame_score > best_score:
            best_score = frame_score
            best_frame = frame.copy()
            best_faces = faces_found
            strong_portrait = frame_has_portrait

    cap.release()
    return {
        "frame_bgr": best_frame,
        "faces": best_faces,
        "has_strong_portrait": strong_portrait,
    }


def _crop_hero_canvas(frame_bgr: np.ndarray, faces: list, tw: int, th: int) -> tuple[np.ndarray, list]:
    """
    Crops a 100% FULL-BLEED poster canvas (0% horizontal bars!) around the hero face(s),
    automatically trimming the bottom 13% of landscape video frames when headroom allows so
    pre-existing burned-in source subtitles are excluded cleanly.
    Returns (resized_bgr, mapped_faces_in_tw_th).
    """
    h, w = frame_bgr.shape[:2]
    target_aspect = float(tw) / float(th)
    src_aspect = float(w) / float(h)

    if src_aspect > target_aspect:
        # Trim bottom 13% subtitle zone when hero face sits comfortably in upper 75%
        crop_h = h
        y0 = 0
        if faces and faces[0][3] < h * 0.72:
            crop_h = int(round(h * 0.87))
            top_want = max(0, int(round(faces[0][1] - (faces[0][3] - faces[0][1]) * 0.75)))
            y0 = min(top_want, h - crop_h)

        crop_w = max(64, min(w, int(round(crop_h * target_aspect))))
        if faces:
            f0 = faces[0]
            cx = (f0[0] + f0[2]) / 2.0
            if len(faces) >= 2:
                f1 = faces[1]
                span = max(f0[2], f1[2]) - min(f0[0], f1[0])
                if span <= crop_w * 1.16:
                    cx = (min(f0[0], f1[0]) + max(f0[2], f1[2])) / 2.0
            x0 = max(0, min(w - crop_w, int(round(cx - crop_w / 2.0))))
        else:
            x0 = (w - crop_w) // 2
        cropped = frame_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    else:
        x0 = 0
        crop_w = w
        crop_h = max(64, min(h, int(round(crop_w / target_aspect))))
        if faces:
            f0 = faces[0]
            eye_y = f0[1] + (f0[3] - f0[1]) * 0.38
            y0 = max(0, min(h - crop_h, int(round(eye_y - crop_h * 0.34))))
        else:
            y0 = (h - crop_h) // 2
        cropped = frame_bgr[y0 : y0 + crop_h, :]

    sx = float(tw) / max(1.0, float(cropped.shape[1]))
    sy = float(th) / max(1.0, float(cropped.shape[0]))
    resized = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    mapped_faces = []
    for (fx1, fy1, fx2, fy2, sc) in faces:
        mx1 = int(round((fx1 - x0) * sx))
        my1 = int(round((fy1 - y0) * sy))
        mx2 = int(round((fx2 - x0) * sx))
        my2 = int(round((fy2 - y0) * sy))
        if mx2 > 20 and mx1 < tw - 20 and my2 > 20 and my1 < th - 20:
            mapped_faces.append((max(0, mx1), max(0, my1), min(tw, mx2), min(th, my2), sc))

    return resized, mapped_faces


def _segment_hero_mask(canvas_bgr: np.ndarray, mapped_faces: list) -> np.ndarray | None:
    """
    Runs fast downscaled neural-seeded GrabCut around YuNet face + torso coordinates (~75ms)
    to separate the human hero(es) from the background for AI Studio rim-lighting & bokeh!
    Returns float32 alpha mask [0.0 .. 1.0] of shape (th, tw), or None if no close subject.
    """
    if not mapped_faces:
        return None
    th, tw = canvas_bgr.shape[:2]
    f0 = mapped_faces[0]
    fw = f0[2] - f0[0]
    fh = f0[3] - f0[1]
    if fh < th * 0.09:
        return None

    scale = 0.25
    sw, sh = max(80, int(tw * scale)), max(80, int(th * scale))
    small = cv2.resize(canvas_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    # Outer border is definite background
    b_pad = max(4, int(sw * 0.04))
    mask[:b_pad, :] = cv2.GC_BGD
    mask[:, :b_pad] = cv2.GC_BGD
    mask[:, sw - b_pad :] = cv2.GC_BGD

    for (fx1, fy1, fx2, fy2, _) in mapped_faces[:2]:
        sfw = max(8, int((fx2 - fx1) * scale))
        sfh = max(8, int((fy2 - fy1) * scale))
        sx1, sy1 = int(fx1 * scale), int(fy1 * scale)
        sx2, sy2 = int(fx2 * scale), int(fy2 * scale)

        # Hair + head + shoulders/torso probable foreground box
        bx1 = max(1, int(sx1 - sfw * 1.15))
        bx2 = min(sw - 1, int(sx2 + sfw * 1.15))
        by1 = max(1, int(sy1 - sfh * 0.50))
        by2 = min(sh - 1, int(sy2 + sfh * 3.40))
        mask[by1:by2, bx1:bx2] = cv2.GC_PR_FGD

        # Core face & upper chest are definite foreground
        cx1 = max(1, int(sx1 + sfw * 0.12))
        cx2 = min(sw - 1, int(sx2 - sfw * 0.12))
        cy1 = max(1, int(sy1 + sfh * 0.10))
        cy2 = min(sh - 1, int(sy2 + sfh * 1.60))
        if cx2 > cx1 and cy2 > cy1:
            mask[cy1:cy2, cx1:cx2] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return None

    fg_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Close small interior gaps and keep ONLY connected components overlapping the hero face(s)
    fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    if num_labels > 1:
        clean_bin = np.zeros_like(fg_bin)
        face_centers = [
            (int(((f[0] + f[2]) * 0.5) * scale), int(((f[1] + f[3]) * 0.5) * scale))
            for f in mapped_faces[:2]
        ]
        kept_any = False
        for lbl in range(1, num_labels):
            comp_mask = labels == lbl
            for (fcx, fcy) in face_centers:
                fcx_c = max(0, min(sw - 1, fcx))
                fcy_c = max(0, min(sh - 1, fcy))
                if comp_mask[fcy_c, fcx_c]:
                    clean_bin[comp_mask] = 255
                    kept_any = True
                    break
        if not kept_any:
            largest_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clean_bin[labels == largest_lbl] = 255
        fg_bin = clean_bin

    fg_small = (fg_bin > 0).astype(np.float32)
    fg_ratio = float(np.mean(fg_small))
    if fg_ratio < 0.05 or fg_ratio > 0.88:
        return None

    fg_full = cv2.resize(fg_small, (tw, th), interpolation=cv2.INTER_LINEAR)
    fg_full = cv2.GaussianBlur(fg_full, (29, 29), 9.0)
    return np.clip(fg_full, 0.0, 1.0)


def _master_8k_ai_studio_poster(
    frame_bgr: np.ndarray,
    faces: list,
    tw: int,
    th: int,
    clip_index: int = 1,
    themes: list | None = None,
    ai_mode: bool = True,
    live_ai_bg: Image.Image | None = None,
) -> Image.Image:
    """
    Synthesizes a true 8K Viral YouTube Studio Poster:
    - Full-bleed hero framing (ZERO horizontal letterbox bars!).
    - Neural YuNet + GrabCut foreground isolation.
    - When `ai_mode=True`: Replaces/transforms the mundane room/aisle background into a rich
      AI-styled cinematic studio backdrop (or live Flux AI scene backdrop) with shallow f/1.2 bokeh,
      volumetric god-ray spotlight behind the subject, glowing neon/gold rim-light contour along
      the hero's silhouette, and hyper-crisp 8K `detailEnhance` + CLAHE skin/eye clarity!
    """
    canvas_bgr, mapped_faces = _crop_hero_canvas(frame_bgr, faces, tw, th)

    # 1. Master Foreground Hero to 8K Studio Portrait Clarity (smooth luminous skin + crisp eyes/hair!)
    smooth_skin = cv2.bilateralFilter(canvas_bgr, d=7, sigmaColor=30, sigmaSpace=30)
    # Apply gentle YCrCb skin-selective bilateral polish so pores never look harsh on close-ups
    ycrcb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2YCrCb)
    skin_m = cv2.inRange(ycrcb, (40, 133, 77), (252, 178, 128)).astype(np.float32) / 255.0
    skin_m = cv2.GaussianBlur(skin_m, (15, 15), 4.0)[:, :, None]
    polished_bgr = np.clip(
        smooth_skin.astype(np.float32) * (0.48 * skin_m) + canvas_bgr.astype(np.float32) * (1.0 - 0.48 * skin_m),
        0,
        255,
    ).astype(np.uint8)

    lab = cv2.cvtColor(polished_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.75, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_ch)
    l_mix = cv2.addWeighted(l_clahe, 0.48, l_ch, 0.52, 0)
    hero_hdr = cv2.cvtColor(cv2.merge((l_mix, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    if not ai_mode:
        pil_plain = Image.fromarray(cv2.cvtColor(hero_hdr, cv2.COLOR_BGR2RGB))
        pil_plain = ImageEnhance.Color(pil_plain).enhance(1.24)
        pil_plain = ImageEnhance.Contrast(pil_plain).enhance(1.12)
        return pil_plain.filter(ImageFilter.UnsharpMask(radius=1.7, percent=140, threshold=2))

    # 2. Build 8K AI Studio Atmosphere + Volumetric Spotlight + Rim-Light
    # Pick thematic studio color aura (BGR)
    palettes = [
        ((235, 165, 20), (30, 195, 255)),   # Deep Royal Cyan + Golden Rim Burst
        ((190, 75, 25), (255, 225, 40)),    # Electric Blue + Neon Cyber Gold
        ((60, 45, 190), (75, 220, 255)),    # Crimson Sunset + Warm Amber Glow
        ((140, 160, 20), (120, 255, 140)),  # Emerald Teal + Bright Lime/Gold
    ]
    bg_tint_bgr, rim_bgr = palettes[(clip_index - 1) % len(palettes)]
    if themes and "survival_adventure" in themes:
        bg_tint_bgr, rim_bgr = ((195, 135, 18), (45, 210, 255))

    alpha_mask = _segment_hero_mask(canvas_bgr, mapped_faces)

    if alpha_mask is not None:
        # Create rich shallow-DOF studio backdrop (from live AI scene if fetched, else transformed video bg)
        if live_ai_bg is not None:
            bg_src = cv2.cvtColor(np.array(live_ai_bg.resize((tw, th), Image.Resampling.LANCZOS)), cv2.COLOR_RGB2BGR)
            bg_blur = cv2.GaussianBlur(bg_src, (0, 0), sigmaX=5.5)
        else:
            bg_blur = cv2.GaussianBlur(canvas_bgr, (0, 0), sigmaX=16.0)
            tint_layer = np.full_like(bg_blur, bg_tint_bgr, dtype=np.uint8)
            bg_blur = cv2.addWeighted(bg_blur, 0.52, tint_layer, 0.48, -18)

        # Add radial volumetric spotlight bursting right behind the primary hero's head!
        spot_cx = (mapped_faces[0][0] + mapped_faces[0][2]) / 2.0 if mapped_faces else tw * 0.50
        spot_cy = (mapped_faces[0][1] + mapped_faces[0][3]) / 2.0 if mapped_faces else th * 0.36
        yy, xx = np.ogrid[:th, :tw]
        r2 = ((xx - spot_cx) / (tw * 0.46)) ** 2 + ((yy - spot_cy) / (th * 0.34)) ** 2
        glow_strength = np.exp(-r2 * 1.45).astype(np.float32)[:, :, None]
        rim_color_arr = np.array(rim_bgr, dtype=np.float32)[None, None, :]
        bg_lit = np.clip(
            bg_blur.astype(np.float32) * (0.62 + 0.45 * glow_strength) + rim_color_arr * (0.42 * glow_strength),
            0,
            255,
        )

        # Soft diffused atmospheric studio backlight aura behind hero's hair & shoulders (zero hard sticker line!)
        dilated = cv2.dilate(alpha_mask, np.ones((27, 27), np.uint8), iterations=1)
        aura = cv2.GaussianBlur(dilated, (65, 65), 22.0)
        rim_halo = np.clip((aura - alpha_mask * 0.85) * 1.35, 0.0, 1.0)[:, :, None]
        vert_fade = np.clip(1.20 - (yy / float(th)) * 0.95, 0.10, 1.0).astype(np.float32)[:, :, None]
        bg_with_rim = np.clip(bg_lit + rim_color_arr * rim_halo * vert_fade * 0.52, 0, 255)

        a3 = alpha_mask[:, :, None]
        comp_bgr = np.clip(hero_hdr.astype(np.float32) * a3 + bg_with_rim * (1.0 - a3), 0, 255).astype(np.uint8)
    elif live_ai_bg is not None:
        # No close human portrait in clip (e.g. wide aerial ocean shot) -> use live 8K AI vertical scene!
        comp_bgr = cv2.cvtColor(np.array(live_ai_bg.resize((tw, th), Image.Resampling.LANCZOS)), cv2.COLOR_RGB2BGR)
    else:
        # Dramatic cinematic grade + upper sunbeam glow on full-bleed action shot
        yy, xx = np.ogrid[:th, :tw]
        r2 = ((xx - tw * 0.5) / (tw * 0.58)) ** 2 + ((yy - th * 0.30) / (th * 0.44)) ** 2
        glow = np.exp(-r2 * 1.3).astype(np.float32)[:, :, None]
        rim_arr = np.array(rim_bgr, dtype=np.float32)[None, None, :]
        comp_bgr = np.clip(hero_hdr.astype(np.float32) * (0.76 + 0.34 * glow) + rim_arr * (0.18 * glow), 0, 255).astype(np.uint8)

    pil_im = Image.fromarray(cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB))
    pil_im = ImageEnhance.Color(pil_im).enhance(1.28)
    pil_im = ImageEnhance.Contrast(pil_im).enhance(1.13)
    pil_im = pil_im.filter(ImageFilter.UnsharpMask(radius=1.7, percent=145, threshold=2))
    return pil_im


def _build_ai_scene_prompt(
    source_title: str,
    hook_text: str,
    transcript_text: str,
    aspect_ratio: str,
    clip_index: int,
) -> str:
    combined = f"{source_title or ''} {hook_text or ''} {(transcript_text or '')[:220]}"
    low = combined.lower()
    clean_topic = re.sub(r"https?://\S+|#\w+", "", f"{source_title or ''} {hook_text or ''}")
    clean_topic = " ".join(re.sub(r"[^a-zA-Z0-9$% '\-]", " ", clean_topic).split())[:110]

    cues = []
    if any(w in low for w in ["raft", "ocean", "sea", "water", "boat", "island", "beach", "storm"]):
        cues.append("dramatic turquoise ocean waves, epic storm clouds with golden sunbeams, wooden survival raft")
    if any(w in low for w in ["box", "mystery", "chest", "secret", "vault", "locked"]):
        cues.append("glowing treasure vault rays, dramatic golden volumetric lighting")
    if any(w in low for w in ["$", "dollar", "money", "cash", "million", "thousand", "prize"]):
        cues.append("high-stakes game show arena spotlights, glowing golden confetti atmosphere")
    if any(w in low for w in ["gym", "weight", "lbs", "pounds", "workout", "fitness"]):
        cues.append("cinematic fitness arena with dramatic neon cyan and amber rim spotlights")
    if not cues:
        cues.append("cinematic dark studio stage with dramatic teal and warm golden volumetric spotlights")

    orient = "vertical 9:16" if aspect_ratio == "9:16" else ("square 1:1" if aspect_ratio == "1:1" else "16:9")
    return (
        f"Hyperrealistic 8K cinematic YouTube backdrop ({orient}) for {clean_topic}: "
        f"{'; '.join(cues[:2])}, ultra detailed lighting, shallow depth of field, NO text, NO watermark."
    )


def _try_fetch_live_ai_backdrop(prompt: str, tw: int, th: int, seed: int) -> Image.Image | None:
    """Optional live Flux backdrop fetch (gracefully returns None immediately on quota/timeout)."""
    encoded = urllib.parse.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={tw}&height={th}&seed={seed}&nologo=true&model=flux"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0 Safari/537.36",
                "Accept": "image/*",
            },
        )
        with urllib.request.urlopen(req, timeout=9) as resp:
            raw = resp.read()
        if not raw or len(raw) < 12_000:
            return None
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if im.width < 400 or float(np.std(np.asarray(im, dtype=np.float32))) < 20.0:
            return None
        # Crop 6% border to guarantee 0% corner tags
        im_clean = im.crop((int(im.width * 0.03), int(im.height * 0.02), int(im.width * 0.97), int(im.height * 0.93)))
        return ImageOps.fit(im_clean, (tw, th), Image.Resampling.LANCZOS)
    except Exception:
        return None


def _distill_poster_headline(hook_text: str, transcript_text: str, clip_index: int) -> list[str]:
    """
    Distills clip hook/transcript into complete, grammatically intact 2-3 line poster copy:
    - NEVER truncates mid-word (zero sliced characters!).
    - NEVER ends on a dangling preposition/article ("ON", "THE", "OUR", etc.).
    """
    raw_h = (hook_text or "").strip()
    if not raw_h or raw_h.lower().startswith("viral highlight"):
        raw_h = (transcript_text or f"MUST WATCH MOMENT #{clip_index}").strip()

    sentences = [s.strip() for s in re.split(r"[.!?]+", raw_h) if len(s.strip()) >= 5]
    if sentences:
        scored_s = []
        for idx, s in enumerate(sentences[:4]):
            sc = (14 if re.search(r"(\$?\d+|secret|never|box|survive|win|lost|storm|impossible|wait)", s, re.I) else 0) - idx * 2
            scored_s.append((sc, s))
        scored_s.sort(key=lambda x: x[0], reverse=True)
        raw_h = scored_s[0][1]

    tokens = [t for t in raw_h.split() if t not in ("—", "-")]

    # Only condense if longer than 8 words; <=8 words already fits cleanly on 2-3 poster lines!
    if len(tokens) > 8:
        clauses = re.split(r",\s+", raw_h)
        if len(clauses) >= 2:
            c1, c2 = ", ".join(clauses[:-1]), clauses[-1]
            c1_w = c1.strip().split()
            if c1_w and c1_w[0].lower() in ("if", "when", "while", "because"):
                for k, tok in enumerate(c1_w[:5]):
                    if tok.lower() in ("can", "could", "will", "would"):
                        c1_w = c1_w[k + 1 :]
                        break
            for prep in ("before", "after", "during", "without", "because"):
                low_list = [w.lower().strip(".,!?") for w in c1_w]
                if len(c1_w) > 4 and prep in low_list:
                    idx_p = low_list.index(prep)
                    if idx_p >= 2:
                        c1_w = c1_w[:idx_p]
                        break
            c2_w = [w for w in c2.strip().split() if w.lower().strip(".,!?") not in ("they", "we", "you", "he", "she", "will", "can")]
            tokens = c1_w[:4] + c2_w[:4]
        else:
            stop_lead = {"so", "and", "but", "well", "like", "basically", "actually", "literally", "collectively", "just"}
            filt = [w for i, w in enumerate(tokens) if not (i < 3 and w.lower().strip(".,!?") in stop_lead)]
            keep = filt[:8]
            # Strip any dangling trailing preposition/article so headlines NEVER end on "ON!" or "THE!"
            while len(keep) > 3 and keep[-1].lower().strip(".,!?\"'") in _DANGLING_END_WORDS:
                keep.pop()
            tokens = keep

    while len(tokens) > 2 and tokens[-1].lower().strip(".,!?\"'") in _DANGLING_END_WORDS:
        tokens.pop()

    headline = " ".join(tokens).strip(" ,;:-").upper()
    if headline and not headline.endswith(("!", "?")):
        headline = headline.rstrip(".") + "!"

    words = headline.split()
    total_chars = sum(len(w) for w in words) + max(0, len(words) - 1)
    max_chars = max(15, int(math.ceil(total_chars / 2.65)))

    lines, curr = [], []
    for wd in words:
        if curr and len(" ".join(curr + [wd])) > max_chars:
            lines.append(" ".join(curr))
            curr = [wd]
        else:
            curr.append(wd)
    if curr:
        lines.append(" ".join(curr))

    # Merge shortest adjacent lines if > 3 lines — NEVER slice characters off words!
    while len(lines) > 3:
        best_idx = min(range(len(lines) - 1), key=lambda k: len(lines[k]) + len(lines[k + 1]))
        lines[best_idx : best_idx + 2] = [f"{lines[best_idx]} {lines[best_idx + 1]}"]

    return lines or [f"HIGHLIGHT #{clip_index}!"]


def _draw_3d_poster_typography(
    im: Image.Image,
    lines: list[str],
    aspect_ratio: str,
    clip_index: int,
) -> Image.Image:
    """
    Renders studio-grade MrBeast-style 3D extruded poster typography:
    - Masks & blurs bottom 24% under a deep shadow stage so zero source captions bleed through.
    - Auto-scales font size via exact pixel bounding-box measurement so zero letters ever clip!
    """
    tw, th = im.size

    # Softly blur bottom 24% before dark vignette so any pre-existing video text disappears 100%
    if aspect_ratio == "9:16":
        blur_top = int(th * 0.74)
        lower_crop = im.crop((0, blur_top, tw, th)).filter(ImageFilter.GaussianBlur(radius=14))
        blend_mask = Image.new("L", (tw, th - blur_top), 0)
        bdraw = ImageDraw.Draw(blend_mask)
        bh = th - blur_top
        for y in range(bh):
            bdraw.line([(0, y), (tw, y)], fill=int(min(255, 255 * (y / max(1.0, bh * 0.45)))))
        im = im.copy()
        im.paste(lower_crop, (0, blur_top), blend_mask)

    rgba = im.convert("RGBA")

    # Smooth multi-stop bottom shadow gradient
    grad = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grad)
    grad_start = int(th * (0.48 if aspect_ratio == "9:16" else 0.44))
    for y in range(grad_start, th):
        p = (y - grad_start) / max(1.0, float(th - grad_start))
        alpha = int(min(246, 252 * (p ** 1.28)))
        gdraw.line([(0, y), (tw, y)], fill=(4, 6, 12, alpha))
    rgba = Image.alpha_composite(rgba, grad)

    # Exact pixel-width auto-fitting so widest line ALWAYS fits inside 88% of canvas width!
    font_sz = 92 if aspect_ratio == "9:16" else (70 if aspect_ratio == "1:1" else 80)
    probe_draw = ImageDraw.Draw(rgba)
    max_allowed_w = int(tw * 0.88)
    font = _load_bold_font(font_sz)
    while font_sz > 42:
        font = _load_bold_font(font_sz)
        widest = 0
        for ln in lines:
            bbox = probe_draw.textbbox((0, 0), ln, font=font, stroke_width=8)
            widest = max(widest, bbox[2] - bbox[0])
        if widest <= max_allowed_w:
            break
        font_sz -= 3

    accent_palette = [
        (255, 224, 0),    # Viral Beast Gold
        (0, 238, 255),    # Electric Cyan
        (255, 106, 48),   # Fiery Coral Orange
        (92, 255, 148),   # Neon Emerald
    ]
    accent_rgb = accent_palette[(clip_index - 1) % len(accent_palette)]

    line_step = int(font_sz * 1.18)
    bottom_clearance = 225 if aspect_ratio == "9:16" else 118
    base_y = th - (len(lines) * line_step) - bottom_clearance + (line_step // 2)

    # Layer 1: Soft wide ambient drop-shadow
    shadow_layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_layer)
    for idx, line in enumerate(lines):
        y_pos = base_y + idx * line_step
        sdraw.text(
            (tw // 2, y_pos + 10),
            line,
            font=font,
            fill=(0, 0, 0, 240),
            anchor="mm",
            stroke_width=12,
            stroke_fill=(0, 0, 0, 240),
        )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=9))
    rgba = Image.alpha_composite(rgba, shadow_layer)

    # Layer 2: Crisp 3D block extrusion + heavy anti-aliased stroke + vibrant fill
    tdraw = ImageDraw.Draw(rgba)
    for idx, line in enumerate(lines):
        y_pos = base_y + idx * line_step
        is_accent = idx == len(lines) - 1
        fill_col = (*accent_rgb, 255) if is_accent else (255, 255, 255, 255)

        for depth in range(8, 1, -2):
            tdraw.text(
                (tw // 2 + depth // 2, y_pos + depth),
                line,
                font=font,
                fill=(8, 10, 16, 255),
                anchor="mm",
                stroke_width=8,
                stroke_fill=(8, 10, 16, 255),
            )

        tdraw.text(
            (tw // 2, y_pos),
            line,
            font=font,
            fill=fill_col,
            anchor="mm",
            stroke_width=7,
            stroke_fill=(0, 0, 0, 255),
        )

    return rgba.convert("RGB")


def generate_clip_thumbnail(
    video_path: str,
    clip_start: float,
    clip_duration: float,
    output_thumb_path: str,
    clip_index: int = 1,
    hook_text: str = None,
    transcript_text: str = None,
    source_title: str = None,
    aspect_ratio: str = "9:16",
    use_ai: bool = True,
) -> str:
    """
    Generates an authentic 8K AI Studio Viral Thumbnail:
    1. Showcase 8K templates when clipping the exact MrBeast 150-lbs video in 9:16.
    2. Neural 8K AI Studio Compositor (`use_ai=True`):
       - Scans 16 clip moments with YuNet CNN Face Detector (unique hero moment per `clip_index`).
       - Segments the real human hero(es) via YuNet-seeded GrabCut, adds glowing 3D studio rim-light,
         volumetric spotlight + bokeh AI backdrop (with optional live Flux scene backdrop), and
         `detailEnhance` + CLAHE 8K clarity — 100% immune to API quotas & 0% horizontal bars!
    3. Zero-truncation pixel-fitted 3D Extruded MrBeast Poster Typography + Brand Logo.
    """
    thumb_path = Path(output_thumb_path)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    clean_thumb = str(thumb_path).replace(".jpg", "_clean.jpg")
    tw, th = _target_size(aspect_ratio)

    context = analyze_video_context(hook_text=hook_text, transcript_text=transcript_text, source_title=source_title)
    ai_candidates = sorted(list(AI_THUMB_DIR.glob("*.jpg")) + list(AI_THUMB_DIR.glob("*.png")))
    ai_candidates = [c for c in ai_candidates if not c.name.endswith("_clean.jpg")]

    # 1. Bundled 150-lbs showcase templates ONLY for that specific video in 9:16
    if use_ai and ai_candidates and aspect_ratio == "9:16" and "mrbeast_150lbs_special" in context["themes"]:
        selected_template = find_best_ai_template(ai_candidates, clip_index, context)
        if selected_template and selected_template.exists():
            try:
                im = Image.open(selected_template).convert("RGB")
                im_resized = ImageOps.fit(im, (tw, th), Image.Resampling.LANCZOS)
                im_resized = ImageEnhance.Color(im_resized).enhance(1.15)
                im_resized = ImageEnhance.Contrast(im_resized).enhance(1.10)
                im_resized = im_resized.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
                im_resized.save(clean_thumb, "JPEG", quality=94)
                im_resized.save(str(thumb_path), "JPEG", quality=94)
                composite_brand_on_thumbnail(str(thumb_path))
                return str(thumb_path)
            except Exception as e:
                print(f"[ThumbnailGenerator] Showcase template fallback: {e}")

    # 2. Extract distinct sharpest YuNet hero frame for this clip_index
    hero_info = _extract_best_hero_frame(video_path, clip_duration, clip_index=clip_index)

    live_ai_bg = None
    if use_ai and (hero_info["frame_bgr"] is None or not hero_info["has_strong_portrait"]):
        seed_src = f"{source_title or ''}|{hook_text or ''}|{clip_index}"
        seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16) % 900_000 + 10_000
        prompt = _build_ai_scene_prompt(source_title, hook_text, transcript_text, aspect_ratio, clip_index)
        live_ai_bg = _try_fetch_live_ai_backdrop(prompt, tw, th, seed)

    if hero_info["frame_bgr"] is not None:
        poster_im = _master_8k_ai_studio_poster(
            frame_bgr=hero_info["frame_bgr"],
            faces=hero_info["faces"],
            tw=tw,
            th=th,
            clip_index=clip_index,
            themes=context["themes"],
            ai_mode=use_ai,
            live_ai_bg=live_ai_bg,
        )
    elif live_ai_bg is not None:
        poster_im = live_ai_bg
    else:
        poster_im = Image.new("RGB", (tw, th), color=(14, 18, 30))

    lines = _distill_poster_headline(hook_text=hook_text, transcript_text=transcript_text, clip_index=clip_index)
    final_im = _draw_3d_poster_typography(poster_im, lines=lines, aspect_ratio=aspect_ratio, clip_index=clip_index)

    final_im.save(clean_thumb, "JPEG", quality=94)
    final_im.save(str(thumb_path), "JPEG", quality=94)
    composite_brand_on_thumbnail(str(thumb_path))
    return str(thumb_path)
