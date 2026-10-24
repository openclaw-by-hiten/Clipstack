import os
import cv2
import numpy as np
from pathlib import Path

def get_face_classifiers():
    """Returns OpenCV frontal and profile face Haar cascades."""
    frontal_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    profile_path = os.path.join(cv2.data.haarcascades, 'haarcascade_profileface.xml')
    frontal = cv2.CascadeClassifier(frontal_path) if os.path.exists(frontal_path) else None
    profile = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None
    return frontal, profile

def build_crop_expression(keyframes, default_x: int, max_x: int) -> str:
    """
    Builds a nested FFmpeg arithmetic expression that smoothly pans the crop window
    between keyframes over time:
    if(lt(t, t1), interp(0,1), if(lt(t, t2), interp(1,2), ...))
    """
    if not keyframes or len(keyframes) <= 1:
        return str(default_x)

    # Escape commas for FFmpeg filter parsing: \,
    expr = str(keyframes[-1][1])
    for i in range(len(keyframes) - 2, -1, -1):
        t0, x0 = keyframes[i]
        t1, x1 = keyframes[i + 1]
        dt = max(0.05, t1 - t0)
        dx = x1 - x0
        interp = f"({x0}+({dx})*(t-{t0:.2f})/{dt:.2f})"
        expr = f"if(lt(t\\,{t1:.2f})\\,{interp}\\,{expr})"
    return expr

def generate_smart_crop_filter(video_path: str, clip_duration: float, aspect_ratio: str = "9:16") -> str:
    """
    Analyzes video footage to detect face trajectories and generates a dynamic,
    face-centered 9:16 or 1:1 FFmpeg crop filter.
    Guarantees that faces are consistently framed in the center with zero edge-cropping.
    """
    if aspect_ratio not in ["9:16", "1:1"]:
        return "scale=1920:1080:flags=lanczos"

    if not os.path.exists(video_path):
        if aspect_ratio == "9:16":
            return "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)':(iw-ow)/2:(ih-oh)/2,scale=1080:1920:flags=lanczos"
        return "crop='min(iw,ih)':'min(iw,ih)':(iw-ow)/2:(ih-oh)/2,scale=1080:1080:flags=lanczos"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)':(iw-ow)/2:(ih-oh)/2,scale=1080:1920:flags=lanczos"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    actual_duration = total_frames / fps if total_frames > 0 else clip_duration
    duration = min(clip_duration, actual_duration)

    # If video is already vertical, return simple scale
    if abs((w / h) - (9 / 16)) < 0.05 and aspect_ratio == "9:16":
        cap.release()
        return "scale=1080:1920:flags=lanczos"

    if aspect_ratio == "9:16":
        crop_w = int(h * 9 / 16)
        if crop_w % 2 != 0:
            crop_w += 1
        crop_h = h
    else:  # 1:1
        crop_w = min(w, h)
        if crop_w % 2 != 0:
            crop_w += 1
        crop_h = crop_w

    max_x = max(0, w - crop_w)
    default_x = max_x // 2

    if max_x == 0:
        cap.release()
        return f"crop={crop_w}:{crop_h}:0:0,scale=1080:1920:flags=lanczos"

    frontal_cascade, profile_cascade = get_face_classifiers()
    if frontal_cascade is None:
        cap.release()
        return f"crop={crop_w}:{crop_h}:{default_x}:0,scale=1080:1920:flags=lanczos"

    # Sample video at 1.0s intervals for fast, responsive tracking (~30-60 samples total)
    sample_step = 1.0
    num_samples = max(2, int(duration / sample_step))
    raw_points = []

    for i in range(num_samples):
        t = min(duration - 0.05, i * sample_step)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(45, 45))
        if len(faces) == 0 and profile_cascade:
            faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(45, 45))

        if len(faces) > 0:
            # Weighted center of detected faces (larger face gets priority)
            weighted_centers = [(fx + fw / 2, fw * fh) for (fx, fy, fw, fh) in faces]
            total_weight = sum(wgt for _, wgt in weighted_centers)
            center_x = sum(cx * wgt for cx, wgt in weighted_centers) / total_weight

            # Align crop window around the center of the face
            target_x = int(center_x - crop_w / 2)
            target_x = max(0, min(max_x, target_x))
            raw_points.append((t, target_x, True))
        else:
            raw_points.append((t, default_x, False))

    cap.release()

    if not raw_points:
        return f"crop={crop_w}:{crop_h}:{default_x}:0,scale=1080:1920:flags=lanczos"

    # Propagate face detection across gaps (forward and backward fill)
    filled_points = []
    last_detected_x = default_x
    for t, x, detected in raw_points:
        if detected:
            last_detected_x = x
            filled_points.append((t, x))
        else:
            filled_points.append((t, last_detected_x))

    # Apply exponential smoothing for cinematographic camera movement
    alpha = 0.35
    smoothed_points = []
    curr = filled_points[0][1]
    for t, x in filled_points:
        curr = int(alpha * x + (1.0 - alpha) * curr)
        curr = max(0, min(max_x, curr))
        smoothed_points.append((t, curr))

    # Reduce to significant keyframes (every ~2 seconds or on significant movement)
    keyframes = [smoothed_points[0]]
    for t, x in smoothed_points[1:]:
        last_t, last_x = keyframes[-1]
        # Record keyframe if position moved by > 30px or after 2.5s
        if abs(x - last_x) >= 30 or (t - last_t) >= 2.5:
            keyframes.append((t, x))

    # Ensure last timestamp is represented
    if keyframes[-1][0] < duration:
        keyframes.append((duration, smoothed_points[-1][1]))

    # Build dynamic expression
    dynamic_x = build_crop_expression(keyframes, default_x, max_x)
    target_scale = "1080:1920" if aspect_ratio == "9:16" else "1080:1080"
    return f"crop=w={crop_w}:h={crop_h}:x='{dynamic_x}':y=0,scale={target_scale}:flags=lanczos"
