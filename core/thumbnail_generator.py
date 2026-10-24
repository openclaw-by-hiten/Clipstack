import os
import subprocess
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
from core.branding import composite_brand_on_thumbnail, get_brand_config

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"

def generate_clip_thumbnail(
    video_path: str,
    clip_start: float,
    clip_duration: float,
    output_thumb_path: str,
    clip_index: int = 1,
    hook_text: str = None,
    use_ai: bool = True
) -> str:
    """
    Generates a unique, high-energy 8K-styled vertical thumbnail extracted directly
    from the actual clip video footage.
    - Scans candidate frames across the clip timeline to find the clearest, most expressive face moment.
    - Crops to 9:16 centered on the detected face so faces are never cut off.
    - Enhances with 8K saturation pop (+24%), contrast curve (+20%), and unsharp detail.
    - Automatically composites the active channel brand logo.
    """
    thumb_path = Path(output_thumb_path)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    clean_thumb = str(thumb_path).replace(".jpg", "_clean.jpg")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback to ffmpeg single frame capture
        cmd = [
            "ffmpeg", "-y", "-ss", str(max(0.5, clip_start + 2.0)),
            "-i", video_path, "-frames:v", "1", "-update", "1", "-q:v", "2",
            str(thumb_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        composite_brand_on_thumbnail(str(thumb_path))
        return str(thumb_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    actual_dur = total_frames / fps if total_frames > 0 else clip_duration
    duration = min(clip_duration, actual_dur)

    # Load Haar Cascades
    frontal_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    profile_path = os.path.join(cv2.data.haarcascades, 'haarcascade_profileface.xml')
    frontal_cascade = cv2.CascadeClassifier(frontal_path) if os.path.exists(frontal_path) else None
    profile_cascade = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None

    # Sample timestamps across the clip
    sample_percentages = [0.08, 0.16, 0.25, 0.35, 0.48, 0.60, 0.72, 0.85]
    sample_times = [clip_start + duration * p for p in sample_percentages]

    best_frame = None
    best_score = -1.0
    best_face_box = None

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if frontal_cascade:
            faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50))
        if len(faces) == 0 and profile_cascade:
            faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50))

        # Sharpness & contrast metrics
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = float(gray.std())
        brightness = float(gray.mean())

        # Penalize overly dark/blown-out frames
        if brightness < 20 or brightness > 240:
            continue

        face_score = 0.0
        target_box = None
        if len(faces) > 0:
            largest = max(faces, key=lambda b: b[2] * b[3])
            face_score = float(largest[2] * largest[3])
            target_box = largest

        # Total energy score
        score = face_score * 2.5 + lap_var * 4.0 + contrast * 8.0
        if score > best_score:
            best_score = score
            best_frame = frame
            best_face_box = target_box

    cap.release()

    if best_frame is None:
        # Direct capture fallback
        cmd = [
            "ffmpeg", "-y", "-ss", str(max(0.5, clip_start + 1.5)),
            "-i", video_path, "-frames:v", "1", "-update", "1", "-q:v", "2",
            str(thumb_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        composite_brand_on_thumbnail(str(thumb_path))
        return str(thumb_path)

    # Convert BGR to RGB PIL Image
    rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)

    # Center crop to 9:16 around detected face
    cur_w, cur_h = im.size
    target_ratio = 9 / 16
    cur_ratio = cur_w / cur_h

    if abs(cur_ratio - target_ratio) > 0.05:
        if cur_ratio > target_ratio:
            # Horizontal crop (e.g. 16:9 to 9:16)
            new_w = int(cur_h * target_ratio)
            if best_face_box is not None:
                fx, fy, fw, fh = best_face_box
                center_x = fx + fw // 2
                crop_x1 = max(0, min(cur_w - new_w, center_x - new_w // 2))
            else:
                crop_x1 = (cur_w - new_w) // 2
            im = im.crop((crop_x1, 0, crop_x1 + new_w, cur_h))
        else:
            # Vertical crop
            new_h = int(cur_w / target_ratio)
            crop_y1 = max(0, (cur_h - new_h) // 2)
            im = im.crop((0, crop_y1, cur_w, crop_y1 + new_h))

    im = im.resize((1080, 1920), Image.Resampling.LANCZOS)

    # 8K Poster Enhancement Pipeline
    enh_color = ImageEnhance.Color(im)
    im = enh_color.enhance(1.24)

    enh_contrast = ImageEnhance.Contrast(im)
    im = enh_contrast.enhance(1.20)

    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))

    im.save(clean_thumb, "JPEG", quality=95)
    im.save(str(thumb_path), "JPEG", quality=95)

    # Composite active brand badge
    composite_brand_on_thumbnail(str(thumb_path))
    return str(thumb_path)
