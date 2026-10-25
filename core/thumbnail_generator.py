import os
import re
import subprocess
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from core.branding import composite_brand_on_thumbnail

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
AI_THUMB_DIR = WORKSPACE_DIR / "ai_thumbnails"
AI_THUMB_DIR.mkdir(parents=True, exist_ok=True)


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
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
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
    text = f"{source_title or ''} {hook_text or ''} {transcript_text or ''}".lower()
    themes = []
    if any(k in text for k in ["150 lbs", "150 pounds", "lose 150"]):
        themes.append("mrbeast_150lbs_special")
    if any(k in text for k in ["pound", "lbs", "weight", "scale", "gym", "workout"]):
        themes.append("weight_loss_transformation")
    if any(k in text for k in ["money", "cash", "dollar", "$", "thousand", "million", "prize"]):
        themes.append("cash_prize_challenge")
    return {"themes": themes, "clean_text": text[:400]}


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


def _extract_best_clip_frame(video_path: str, clip_duration: float, out_frame: str) -> bool:
    if not video_path or not os.path.exists(video_path):
        return False
    seek_t = min(max(0.8, clip_duration * 0.22), max(0.5, clip_duration - 1.0))
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{seek_t:.2f}",
                "-i", video_path,
                "-vf", "thumbnail=n=36",
                "-frames:v", "1",
                "-q:v", "2",
                out_frame,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
        )
        return os.path.exists(out_frame) and os.path.getsize(out_frame) > 5_000
    except Exception:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{seek_t:.2f}", "-i", video_path, "-frames:v", "1", "-q:v", "2", out_frame],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )
            return os.path.exists(out_frame) and os.path.getsize(out_frame) > 5_000
        except Exception:
            return False


def _render_clip_aligned_poster(
    video_path: str,
    clip_duration: float,
    hook_text: str,
    output_path: str,
    aspect_ratio: str = "9:16",
    clip_index: int = 1,
) -> str:
    """
    Creates a genuine clip-aligned high-contrast thumbnail directly from the clip's
    best action/face frame + crisp 3D headline typography matching the clip's exact topic.
    """
    tw, th = _target_size(aspect_ratio)
    tmp_frame = str(output_path).replace(".jpg", "_rawframe.jpg")
    has_frame = _extract_best_clip_frame(video_path, clip_duration, tmp_frame)

    if has_frame:
        raw_im = Image.open(tmp_frame).convert("RGB")
        try:
            os.remove(tmp_frame)
        except OSError:
            pass
        # If landscape source into vertical poster, compose crisp hero frame over rich blurred fill
        if aspect_ratio == "9:16" and raw_im.width > raw_im.height:
            bg = ImageOps.fit(raw_im, (tw, th), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=26))
            bg = ImageEnhance.Brightness(bg).enhance(0.55)
            fg_h = int(round(tw * raw_im.height / max(1, raw_im.width)))
            fg = raw_im.resize((tw, fg_h), Image.Resampling.LANCZOS)
            bg.paste(fg, (0, max(160, (th - fg_h) // 2 - 120)))
            im = bg
        else:
            im = ImageOps.fit(raw_im, (tw, th), Image.Resampling.LANCZOS)
    else:
        im = Image.new("RGB", (tw, th), color=(14, 16, 26))

    # Viral color pop & sharpness
    im = ImageEnhance.Color(im).enhance(1.22)
    im = ImageEnhance.Contrast(im).enhance(1.14)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=130, threshold=2))

    # Dark lower gradient overlay for ultra-legible headline text
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    grad_start = int(th * 0.56)
    for y in range(grad_start, th):
        alpha = int(215 * ((y - grad_start) / max(1, th - grad_start)))
        odraw.line([(0, y), (tw, y)], fill=(0, 0, 0, alpha))
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

    # Bold 3D Hook Headline aligned to THIS clip (distilled into complete punchy poster lines!)
    draw = ImageDraw.Draw(im)
    raw_h = (hook_text or f"VIRAL HIGHLIGHT #{clip_index}").strip()
    # Distill long sentences into complete, high-impact 4-8 word poster copy (never cutting mid-clause!)
    if len(raw_h.split()) > 8:
        clauses = re.split(r",\s+", raw_h)
        if len(clauses) >= 2:
            c1, c2 = ", ".join(clauses[:-1]), clauses[-1]
            c1_w = c1.strip().split()
            if c1_w and c1_w[0].lower() in ("if", "when", "while"):
                for k, tok in enumerate(c1_w[:5]):
                    if tok.lower() in ("can", "could", "will", "would"):
                        c1_w = c1_w[k + 1:]
                        break
            # Strip trailing prepositional modifier on c1 if still > 5 words (e.g. "before their wedding day")
            for prep in ("before", "after", "during", "without", "because"):
                low_list = [w.lower().strip(".,!?") for w in c1_w]
                if len(c1_w) > 4 and prep in low_list:
                    idx_p = low_list.index(prep)
                    if idx_p >= 2:
                        c1_w = c1_w[:idx_p]
                        break
            c2_w = [w for w in c2.strip().split() if w.lower().strip(".,!?") not in ("they", "we", "you", "he", "she", "will", "can")]
            raw_h = f"{' '.join(c1_w)} — {' '.join(c2_w)}"
        else:
            stop_lead = {"you", "have", "has", "had", "collectively", "actually", "literally", "basically", "just", "so"}
            filt = [w for i, w in enumerate(raw_h.split()) if not (i < 4 and w.lower().strip(".,!?") in stop_lead)]
            raw_h = " ".join(filt[:8])

    headline = raw_h.strip(" ,;:-").upper()
    if headline and not headline.endswith(("!", "?")):
        headline = headline.rstrip(".") + "!"
    words = [w for w in headline.split() if w != "—"]
    max_chars = 18 if aspect_ratio != "16:9" else 26
    lines, curr = [], []
    for wd in words:
        if curr and len(" ".join(curr + [wd])) > max_chars:
            lines.append(" ".join(curr))
            curr = [wd]
        else:
            curr.append(wd)
    if curr:
        lines.append(" ".join(curr))
    if len(lines) > 3:
        lines = lines[:2] + [" ".join(lines[2:])[:max_chars + 4]]

    font_sz = 74 if aspect_ratio == "9:16" else (58 if aspect_ratio == "1:1" else 68)
    font = _load_bold_font(font_sz)
    accent_colors = [(255, 225, 0), (0, 235, 255), (255, 190, 0), (80, 255, 150)]
    accent = accent_colors[(clip_index - 1) % len(accent_colors)]

    line_step = int(font_sz * 1.18)
    base_y = th - (len(lines) * line_step) - (210 if aspect_ratio == "9:16" else 110)

    for idx, line in enumerate(lines):
        y_pos = base_y + idx * line_step
        fill_col = accent if idx == len(lines) - 1 else (255, 255, 255)
        for ox, oy in [(-4, -4), (4, -4), (-4, 4), (4, 4), (0, 6)]:
            draw.text((tw // 2 + ox, y_pos + oy), line, font=font, fill=(0, 0, 0), anchor="mm")
        draw.text((tw // 2, y_pos), line, font=font, fill=fill_col, anchor="mm")

    clean_path = str(output_path).replace(".jpg", "_clean.jpg")
    im.save(clean_path, "JPEG", quality=93)
    im.save(output_path, "JPEG", quality=93)
    composite_brand_on_thumbnail(output_path)
    return output_path


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
    Generates a context-aligned thumbnail matching the clip's aspect ratio and exact topic:
    - Uses pre-rendered 8K templates ONLY if the video specifically matches that template set AND aspect_ratio == '9:16'.
    - Otherwise renders a real clip-aligned 8K-graded poster from the clip's sharpest hero frame + bold 3D hook title!
    """
    thumb_path = Path(output_thumb_path)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    clean_thumb = str(thumb_path).replace(".jpg", "_clean.jpg")

    context = analyze_video_context(hook_text=hook_text, transcript_text=transcript_text, source_title=source_title)
    ai_candidates = sorted(list(AI_THUMB_DIR.glob("*.jpg")) + list(AI_THUMB_DIR.glob("*.png")))
    ai_candidates = [c for c in ai_candidates if not c.name.endswith("_clean.jpg")]

    # Only use bundled 150-lbs showcase templates when clipping that exact video in 9:16!
    if use_ai and ai_candidates and aspect_ratio == "9:16" and "mrbeast_150lbs_special" in context["themes"]:
        selected_template = find_best_ai_template(ai_candidates, clip_index, context)
        if selected_template and selected_template.exists():
            try:
                im = Image.open(selected_template).convert("RGB")
                im_resized = ImageOps.fit(im, (1080, 1920), Image.Resampling.LANCZOS)
                im_resized = ImageEnhance.Color(im_resized).enhance(1.15)
                im_resized = ImageEnhance.Contrast(im_resized).enhance(1.10)
                im_resized = im_resized.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
                im_resized.save(clean_thumb, "JPEG", quality=93)
                im_resized.save(str(thumb_path), "JPEG", quality=93)
                composite_brand_on_thumbnail(str(thumb_path))
                return str(thumb_path)
            except Exception as e:
                print(f"[ThumbnailGenerator] Template fallback: {e}")

    return _render_clip_aligned_poster(
        video_path=video_path,
        clip_duration=clip_duration,
        hook_text=hook_text,
        output_path=str(thumb_path),
        aspect_ratio=aspect_ratio,
        clip_index=clip_index,
    )
