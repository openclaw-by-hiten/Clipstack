import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_FALLBACK_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

_CASCADES_CACHE = None


def _ensure_yunet_model() -> str | None:
    """Ensures the lightweight 227KB OpenCV YuNet deep face detector ONNX model exists."""
    if YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size > 100_000:
        return str(YUNET_MODEL_PATH)
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_FALLBACK_URL, str(YUNET_MODEL_PATH))
        if YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size > 100_000:
            return str(YUNET_MODEL_PATH)
    except Exception as e:
        print(f"[SmartCrop] YuNet download fallback warning: {e}")
    return None


def create_yunet_detector(input_w: int, input_h: int, score_threshold: float = 0.48):
    """Creates a YuNet FaceDetectorYN instance sized to (input_w, input_h)."""
    if not hasattr(cv2, "FaceDetectorYN"):
        return None
    model_path = _ensure_yunet_model()
    if not model_path:
        return None
    try:
        return cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(int(input_w), int(input_h)),
            score_threshold=float(score_threshold),
            nms_threshold=0.35,
            top_k=200,
        )
    except Exception:
        return None


def _get_cascades():
    """Loads fallback OpenCV Haar cascades for frontal face, profile face, and upper body."""
    global _CASCADES_CACHE
    if _CASCADES_CACHE is not None:
        return _CASCADES_CACHE
    base = cv2.data.haarcascades
    frontal_p = os.path.join(base, "haarcascade_frontalface_default.xml")
    profile_p = os.path.join(base, "haarcascade_profileface.xml")
    upper_p = os.path.join(base, "haarcascade_upperbody.xml")

    frontal = cv2.CascadeClassifier(frontal_p) if os.path.exists(frontal_p) else None
    profile = cv2.CascadeClassifier(profile_p) if os.path.exists(profile_p) else None
    upper = cv2.CascadeClassifier(upper_p) if os.path.exists(upper_p) else None
    _CASCADES_CACHE = (frontal, profile, upper)
    return _CASCADES_CACHE


def _merge_subjects(subjects: list) -> list:
    """
    Merges overlapping subject detections while preserving richest metadata.
    Each subject dict:
      {
        "x1": int, "y1": int, "x2": int, "y2": int,
        "weight": float, "is_face": bool, "conf": float,
        "activity": float, "facing": int (-1 left, 0 front, +1 right)
      }
    """
    if not subjects:
        return []
    ordered = sorted(subjects, key=lambda s: (1 if s["is_face"] else 0, s["weight"]), reverse=True)
    kept = []
    for sub in ordered:
        x1, y1, x2, y2 = sub["x1"], sub["y1"], sub["x2"], sub["y2"]
        cx_a = (x1 + x2) / 2.0
        w_a = max(1, x2 - x1)
        area_a = max(1, w_a * max(1, y2 - y1))
        merged = False

        for kb in kept:
            kx1, ky1, kx2, ky2 = kb["x1"], kb["y1"], kb["x2"], kb["y2"]
            cx_b = (kx1 + kx2) / 2.0
            w_b = max(1, kx2 - kx1)
            area_b = max(1, w_b * max(1, ky2 - ky1))

            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            if inter / min(area_a, area_b) > 0.32 or abs(cx_a - cx_b) < min(w_a, w_b) * 0.58:
                if kb["is_face"] and not sub["is_face"]:
                    kb["activity"] = max(kb["activity"], sub["activity"])
                    merged = True
                    break
                if sub["is_face"] and not kb["is_face"]:
                    kb.update(sub)
                    merged = True
                    break
                kb["x1"] = min(x1, kx1)
                kb["y1"] = min(y1, ky1)
                kb["x2"] = max(x2, kx2)
                kb["y2"] = max(y2, ky2)
                kb["weight"] = max(sub["weight"], kb["weight"])
                kb["conf"] = max(sub["conf"], kb["conf"])
                kb["activity"] = max(sub["activity"], kb["activity"])
                merged = True
                break
        if not merged:
            kept.append(dict(sub))
    return kept


def _compute_subject_activity(diff_full: np.ndarray | None, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> float:
    """
    Measures real-time speaking/lip-motion and hand-gesture energy for a person between
    adjacent frames (~0.08s apart) so active speakers win camera focus in 2-person shots.
    """
    if diff_full is None:
        return 1.0
    fw = max(8, x2 - x1)
    fh = max(8, y2 - y1)

    # Lower-face / jaw / mouth ROI
    mx1 = max(0, int(x1 + fw * 0.14))
    mx2 = min(w, int(x2 - fw * 0.14))
    my1 = max(0, int(y1 + fh * 0.48))
    my2 = min(h, int(y2 + fh * 0.16))
    mouth_e = float(np.mean(diff_full[my1:my2, mx1:mx2])) if (mx2 > mx1 and my2 > my1) else 0.0

    # Upper-body & hand gesture ROI below/around the face
    gx1 = max(0, int(x1 - fw * 0.65))
    gx2 = min(w, int(x2 + fw * 0.65))
    gy1 = max(0, int(y2 - fh * 0.10))
    gy2 = min(h, int(y2 + fh * 2.10))
    gest_e = float(np.mean(diff_full[gy1:gy2, gx1:gx2])) if (gx2 > gx1 and gy2 > gy1) else 0.0

    # Head nod / expression ROI
    head_e = float(np.mean(diff_full[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])) if (x2 > x1 and y2 > y1) else 0.0

    score = 1.0 + 0.30 * mouth_e + 0.22 * gest_e + 0.12 * head_e
    return float(min(4.5, max(1.0, score)))


def _detect_active_side_person(
    small_bgr: np.ndarray,
    diff_small: np.ndarray | None,
    existing_faces: list,
    inv_scale: float,
    orig_w: int,
    orig_h: int,
) -> list:
    """
    Recovers a 2nd person in profile/baseball-cap/turned-head conversation when YuNet sees
    1 primary face, by combining YCrCb skin-tone presence + localized speaking/gesture motion
    outside the primary person's column (ignoring blue water, sky, and foliage).
    """
    if diff_small is None or len(existing_faces) != 1:
        return []

    sh, sw = small_bgr.shape[:2]
    ycrcb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb)
    # Human skin chroma band in YCrCb (excludes ocean blue, grass green, neutral grey)
    skin_mask = cv2.inRange(ycrcb, (45, 134, 78), (250, 178, 126))
    skin_norm = (skin_mask > 0).astype(np.float32)

    # Mask out already-detected person's full body column
    f0 = existing_faces[0]
    fx1_s = int(f0["x1"] / inv_scale)
    fx2_s = int(f0["x2"] / inv_scale)
    fw_s = max(14, fx2_s - fx1_s)
    mask_l = max(0, int(fx1_s - fw_s * 1.15))
    mask_r = min(sw, int(fx2_s + fw_s * 1.15))

    roi_y1, roi_y2 = int(sh * 0.10), int(sh * 0.82)
    if roi_y2 <= roi_y1 + 10:
        return []

    motion_band = diff_small[roi_y1:roi_y2, :].astype(np.float32)
    skin_band = skin_norm[roi_y1:roi_y2, :]

    bg_median = float(np.median(motion_band)) + 0.5
    col_motion = np.mean(motion_band, axis=0)
    col_skin = np.mean(skin_band, axis=0)

    # Suppress primary person's zone
    col_motion[mask_l:mask_r] = 0.0
    col_skin[mask_l:mask_r] = 0.0

    # Smooth over a ~10% frame-width human-head/shoulder window
    win = max(12, int(sw * 0.10))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    smooth_m = np.convolve(col_motion, kernel, mode="same")
    smooth_s = np.convolve(col_skin, kernel, mode="same")

    # Combined active-human presence: requires BOTH real localized motion AND human skin pixels
    combined = smooth_m * (0.25 + 4.0 * smooth_s)
    best_x = int(np.argmax(combined))

    if (
        smooth_m[best_x] >= max(4.2, bg_median * 2.1)
        and smooth_s[best_x] >= 0.045
        and abs(best_x * inv_scale - (f0["x1"] + f0["x2"]) / 2.0) >= orig_w * 0.19
    ):
        # Estimate vertical extent of skin+motion in that column strip
        strip_l = max(0, best_x - win // 2)
        strip_r = min(sw, best_x + win // 2)
        row_act = np.mean(motion_band[:, strip_l:strip_r] * (0.3 + 3.0 * skin_band[:, strip_l:strip_r]), axis=1)
        active_rows = np.where(row_act > np.max(row_act) * 0.28)[0]
        top_s = roi_y1 + (int(active_rows[0]) if len(active_rows) > 0 else int(sh * 0.14))

        est_fw = int(max(orig_h * 0.10, (f0["x2"] - f0["x1"]) * 0.90))
        cx_orig = int(best_x * inv_scale)
        x1 = max(0, cx_orig - est_fw // 2)
        x2 = min(orig_w, cx_orig + est_fw // 2)
        y1 = max(0, int(top_s * inv_scale))
        y2 = min(orig_h, y1 + int(est_fw * 1.18))
        act = min(4.2, 1.2 + 0.30 * float(smooth_m[best_x]))
        facing = -1 if cx_orig > (f0["x1"] + f0["x2"]) / 2.0 else 1
        return [{
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "weight": float(est_fw * est_fw * 1.35),
            "is_face": True,
            "conf": 0.56,
            "activity": act,
            "facing": facing,
        }]
    return []


def detect_subjects_in_frame(
    frame_bgr: np.ndarray,
    next_frame_bgr: np.ndarray | None = None,
    yunet=None,
    yunet_size: tuple[int, int] | None = None,
) -> tuple[list, dict]:
    """
    Detects human faces & active speakers in a single BGR frame using YuNet CNN +
    profile/upper-body fallbacks + adjacent-frame speaking/gesture energy.
    Returns (subjects_list, frame_metrics_dict).
    """
    h, w = frame_bgr.shape[:2]
    det_w = min(w, 960)
    scale = det_w / max(1.0, float(w))
    det_h = max(32, int(round(h * scale)))
    inv_scale = 1.0 / scale

    small = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    diff_small = None
    diff_full = None
    left_e = right_e = mid_e = 0.0
    if next_frame_bgr is not None and next_frame_bgr.shape == frame_bgr.shape:
        small2 = cv2.resize(next_frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        diff_small = cv2.absdiff(gray, gray2)
        diff_full = cv2.resize(diff_small, (w, h), interpolation=cv2.INTER_LINEAR)
        left_e = float(np.mean(diff_small[:, : int(det_w * 0.24)]))
        right_e = float(np.mean(diff_small[:, int(det_w * 0.76) :]))
        mid_e = float(np.mean(diff_small[:, int(det_w * 0.24) : int(det_w * 0.76)])) + 1e-4

    raw_subjects = []

    # 1. Primary Deep CNN Face Detection (YuNet) — catches caps, profiles, glasses, expressions
    active_yunet = yunet
    if active_yunet is None or yunet_size != (det_w, det_h):
        active_yunet = create_yunet_detector(det_w, det_h, score_threshold=0.48)

    if active_yunet is not None:
        active_yunet.setInputSize((det_w, det_h))
        _, faces = active_yunet.detect(small)
        if faces is not None:
            for row in faces:
                fx, fy, fw, fh = row[0], row[1], row[2], row[3]
                conf = float(row[14])
                if fw < det_h * 0.045 or fh < det_h * 0.045:
                    continue
                x1 = max(0, int(round(fx * inv_scale)))
                y1 = max(0, int(round(fy * inv_scale)))
                x2 = min(w, int(round((fx + fw) * inv_scale)))
                y2 = min(h, int(round((fy + fh) * inv_scale)))

                # Infer facing direction (-1 left, 0 front, +1 right) from YuNet eye/nose landmarks
                re_x, le_x, nose_x = float(row[4]), float(row[6]), float(row[8])
                eye_mid = (re_x + le_x) / 2.0
                facing = 0
                if nose_x > eye_mid + fw * 0.07:
                    facing = 1
                elif nose_x < eye_mid - fw * 0.07:
                    facing = -1

                act = _compute_subject_activity(diff_full, x1, y1, x2, y2, w, h)
                area_w = float((x2 - x1) * (y2 - y1)) * (1.25 + 0.75 * conf)
                raw_subjects.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "weight": area_w,
                    "is_face": True,
                    "conf": conf,
                    "activity": act,
                    "facing": facing,
                })

    # 2. Haar Profile / Upper-Body fallback when YuNet finds < 2 faces
    frontal, profile, upper = _get_cascades()
    min_face = max(22, int(det_h * 0.075))
    if len(raw_subjects) == 0 and frontal is not None:
        det_f = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face))
        for (fx, fy, fw, fh) in det_f:
            x1, y1 = int(fx * inv_scale), int(fy * inv_scale)
            x2, y2 = int((fx + fw) * inv_scale), int((fy + fh) * inv_scale)
            act = _compute_subject_activity(diff_full, x1, y1, x2, y2, w, h)
            raw_subjects.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "weight": float((x2 - x1) * (y2 - y1) * 1.5),
                "is_face": True, "conf": 0.60, "activity": act, "facing": 0,
            })

    if len(raw_subjects) < 2 and profile is not None:
        det_p = profile.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face))
        for (fx, fy, fw, fh) in det_p:
            x1, y1 = int(fx * inv_scale), int(fy * inv_scale)
            x2, y2 = int((fx + fw) * inv_scale), int((fy + fh) * inv_scale)
            act = _compute_subject_activity(diff_full, x1, y1, x2, y2, w, h)
            raw_subjects.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "weight": float((x2 - x1) * (y2 - y1) * 1.3),
                "is_face": True, "conf": 0.55, "activity": act, "facing": -1,
            })
        gray_flip = cv2.flip(gray, 1)
        det_pf = profile.detectMultiScale(gray_flip, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face))
        for (fx, fy, fw, fh) in det_pf:
            orig_fx = det_w - fx - fw
            x1, y1 = int(orig_fx * inv_scale), int(fy * inv_scale)
            x2, y2 = int((orig_fx + fw) * inv_scale), int((fy + fh) * inv_scale)
            act = _compute_subject_activity(diff_full, x1, y1, x2, y2, w, h)
            raw_subjects.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "weight": float((x2 - x1) * (y2 - y1) * 1.3),
                "is_face": True, "conf": 0.55, "activity": act, "facing": 1,
            })

    merged = _merge_subjects(raw_subjects)

    # 3. Active Side-Speaker Recovery (catches turned/capped 2nd speaker actively gesturing/talking)
    if len(merged) == 1:
        extra = _detect_active_side_person(small, diff_small, merged, inv_scale, w, h)
        if extra:
            merged = _merge_subjects(merged + extra)

    # 4. Upper-body fallback when zero faces are visible
    if len(merged) == 0 and upper is not None:
        min_ub = max(40, int(det_h * 0.16))
        det_u = upper.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(min_ub, min_ub))
        for (ux, uy, uw, uh) in det_u:
            x1, y1 = int(ux * inv_scale), int(uy * inv_scale)
            x2, y2 = int((ux + uw) * inv_scale), int((uy + uh) * inv_scale)
            act = _compute_subject_activity(diff_full, x1, y1, x2, y2, w, h)
            merged.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "weight": float((x2 - x1) * (y2 - y1) * 0.8),
                "is_face": False, "conf": 0.45, "activity": act, "facing": 0,
            })
        merged = _merge_subjects(merged)

    side_hit = (left_e > 6.5 and right_e > 6.5) or (
        (left_e + right_e) / 2.0 > mid_e * 0.88 and (left_e + right_e) > 11.5
    )
    metrics = {
        "gray_small": gray,
        "side_hit": bool(side_hit),
    }
    return merged, metrics


def build_crop_expression(keyframes: list, default_x: int) -> str:
    """Builds a smooth piecewise-linear FFmpeg crop x expression from (t, x) keyframes."""
    if not keyframes or len(keyframes) <= 1:
        return str(int(keyframes[0][1]) if keyframes else int(default_x))

    # Cap keyframe depth to keep FFmpeg expression evaluation instant and recursion-safe
    if len(keyframes) > 24:
        step = max(1, len(keyframes) // 22)
        reduced = keyframes[::step]
        if reduced[-1][0] != keyframes[-1][0]:
            reduced.append(keyframes[-1])
        keyframes = reduced

    expr = str(int(keyframes[-1][1]))
    for i in range(len(keyframes) - 2, -1, -1):
        t0, x0 = keyframes[i]
        t1, x1 = keyframes[i + 1]
        if abs(x1 - x0) <= 4:
            expr = f"if(lt(t\\,{t1:.2f})\\,{int(x0)}\\,{expr})"
        else:
            dt = max(0.05, t1 - t0)
            dx = x1 - x0
            interp = f"({int(x0)}+({int(dx)})*(t-{t0:.2f})/{dt:.2f})"
            expr = f"if(lt(t\\,{t1:.2f})\\,{interp}\\,{expr})"
    return expr


def _smooth_trajectory(
    raw_points: list,
    default_x: int,
    max_x: int,
    duration: float,
    deadzone_px: int = 34,
) -> list:
    """
    Produces cinema-grade camera keyframes from raw_points: (t, target_x, detected, scene_cut).
    - Holds rock-steady inside deadzone_px (zero micro-jitter).
    - Snaps cleanly across hard scene cuts so the camera never drags across empty walls.
    - Glides smoothly on in-shot active-speaker turn-taking pans.
    """
    if not raw_points:
        return [(0.0, default_x), (duration, default_x)]

    first_valid = next((pt[1] for pt in raw_points if pt[2]), default_x)
    filled = []
    last_x = first_valid
    for pt in raw_points:
        t = pt[0]
        x = pt[1]
        det = pt[2]
        cut = pt[3] if len(pt) > 3 else False
        if det:
            last_x = x
        filled.append((t, last_x, cut))

    smoothed = []
    cam_x = float(filled[0][1])
    anchor_x = cam_x
    for idx, (t, target_x, is_cut) in enumerate(filled):
        if is_cut and abs(target_x - cam_x) > deadzone_px * 1.4:
            # Hard camera edit in source video -> cut cleanly instead of slow sweep!
            if smoothed and t > 0.08:
                prev_x = int(round(cam_x))
                smoothed.append((max(0.0, t - 0.05), max(0, min(max_x, prev_x))))
            anchor_x = float(target_x)
            cam_x = float(target_x)
        else:
            if abs(target_x - anchor_x) > deadzone_px:
                anchor_x = float(target_x)
            alpha = 0.48 if abs(anchor_x - cam_x) > 120 else 0.36
            cam_x = alpha * anchor_x + (1.0 - alpha) * cam_x

        clamped = max(0, min(max_x, int(round(cam_x))))
        smoothed.append((t, clamped))

    keyframes = [smoothed[0]]
    for t, x in smoothed[1:]:
        last_t, last_kx = keyframes[-1]
        if abs(x - last_kx) >= 18 or (t - last_t) >= 2.2:
            keyframes.append((t, x))

    if keyframes[-1][0] < duration:
        keyframes.append((duration, smoothed[-1][1]))
    return keyframes


def analyze_video_subjects(video_path: str, clip_duration: float) -> dict:
    """
    Samples frames at ~0.85s intervals across the clip using YuNet deep face detection +
    adjacent-frame lip/gesture activity + hard scene-cut detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"valid": False}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    actual_dur = total_frames / fps if total_frames > 0 else clip_duration
    duration = max(1.0, min(clip_duration, actual_dur))

    det_w = min(w, 960)
    det_h = max(32, int(round(h * (det_w / max(1.0, float(w))))))
    yunet = create_yunet_detector(det_w, det_h, score_threshold=0.48)

    num_samples = min(46, max(10, int(round(duration / 0.85))))
    frames_data = []
    prev_sample_gray = None
    side_motion_hits = 0

    for i in range(num_samples):
        t = min(duration - 0.10, (i + 0.25) * (duration / num_samples))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Grab adjacent frame ~0.07s later sequentially (fast, zero seek overhead) for lip/gesture motion
        cap.grab()
        ret2, next_frame = cap.read()
        if not ret2:
            next_frame = None

        subjects, metrics = detect_subjects_in_frame(
            frame,
            next_frame_bgr=next_frame,
            yunet=yunet,
            yunet_size=(det_w, det_h),
        )

        gray_s = metrics["gray_small"]
        scene_cut = False
        if prev_sample_gray is not None and prev_sample_gray.shape == gray_s.shape:
            luma_jump = float(np.mean(cv2.absdiff(gray_s, prev_sample_gray)))
            if luma_jump > 33.0:
                scene_cut = True
        prev_sample_gray = gray_s

        if metrics["side_hit"]:
            side_motion_hits += 1

        frames_data.append({
            "t": t,
            "subjects": subjects,
            "scene_cut": scene_cut,
        })

    cap.release()
    side_motion_ratio = side_motion_hits / max(1, len(frames_data))
    return {
        "valid": True,
        "w": w,
        "h": h,
        "duration": duration,
        "frames": frames_data,
        "side_motion_ratio": side_motion_ratio,
    }


def _apply_face_border_guard(crop_x: int, crop_w: int, max_x: int, primary: dict, others: list) -> int:
    """
    Ensures [crop_x, crop_x + crop_w] keeps 100% of `primary` inside with clean breathing room
    while avoiding awkward 'barely-sliced nose tip' (<35% face width) on a secondary person:
    either nudges slightly to show a clear intentional glimpse (>=42% face) or clears the edge.
    """
    outer_pad = max(18, int(crop_w * 0.038))
    min_valid_x = max(0, primary["x2"] + outer_pad - crop_w)
    max_valid_x = min(max_x, primary["x1"] - outer_pad)
    if min_valid_x > max_valid_x:
        # Extreme close-up wider than crop_w: center primary face
        pcx = (primary["x1"] + primary["x2"]) / 2.0
        return max(0, min(max_x, int(round(pcx - crop_w / 2.0))))

    crop_x = max(min_valid_x, min(max_valid_x, crop_x))
    crop_r = crop_x + crop_w

    for s in others:
        sx1, sx2 = s["x1"], s["x2"]
        sfw = max(12, sx2 - sx1)
        # Right border cuts a tiny sliver (<38%) of a right-side person -> try pulling right for >=45% glimpse!
        if sx1 < crop_r < sx2:
            visible_frac = (crop_r - sx1) / float(sfw)
            if visible_frac < 0.42:
                want_glimpse_x = int(round(sx1 + sfw * 0.50 - crop_w))
                if want_glimpse_x <= max_valid_x:
                    crop_x = max(min_valid_x, want_glimpse_x)
                elif (sx1 - 14 - crop_w) >= min_valid_x and visible_frac < 0.16:
                    crop_x = max(min_valid_x, sx1 - 14 - crop_w)
        # Left border cuts a tiny sliver (<38%) of a left-side person -> try pulling left for >=45% glimpse!
        elif sx1 < crop_x < sx2:
            visible_frac = (sx2 - crop_x) / float(sfw)
            if visible_frac < 0.42:
                want_glimpse_x = int(round(sx2 - sfw * 0.50))
                if want_glimpse_x >= min_valid_x:
                    crop_x = min(max_valid_x, want_glimpse_x)
                elif (sx2 + 14) <= max_valid_x and visible_frac < 0.16:
                    crop_x = min(max_valid_x, sx2 + 14)

    return max(0, min(max_x, int(crop_x)))


def _plan_face_track_trajectory(analysis: dict, crop_w: int) -> dict:
    """
    Studio-grade 1- & 2-Person Conversational Gimbal Planner:
    1. Solo Person: Smooth rule-of-thirds / gaze lead-room framing.
    2. Close 2-Person Shot (span <= crop_w): Frames BOTH people 100% simultaneously.
    3. Medium 2-Person Conversation (span <= 1.32 * crop_w):
       Eliminates dead background behind Primary Speaker so Primary is 100% visible AND
       a clear 45%-95% profile/reaction GLIMPSE of Person 2 fits inside the 9:16 frame!
    4. Wide 2-Person Turn-Taking (span > 1.32 * crop_w):
       Tracks Active Speaker (mouth/jaw + gesture motion energy) with ~1.8s shot-hold
       hysteresis so the camera smoothly pans to Person 2 whenever Person 2 speaks/gestures!
    """
    w = analysis["w"]
    duration = analysis["duration"]
    frames = analysis["frames"]
    max_x = max(0, w - crop_w)
    default_x = max_x // 2

    raw_pts = []
    detected_cnt = 0
    three_plus_cnt = 0
    unresolvable_cnt = 0

    active_side = None  # "left" or "right" in 2-speaker scenes
    last_switch_t = -10.0
    prev_cx = w / 2.0

    for fd in frames:
        t = fd["t"]
        is_cut = fd.get("scene_cut", False)
        subs = fd["subjects"]

        if not subs:
            raw_pts.append((t, default_x, False, is_cut))
            continue

        detected_cnt += 1
        # Keep prominent subjects (>= 24% of largest subject weight)
        top_wgt = max(s["weight"] for s in subs)
        prominent = [s for s in subs if s["weight"] >= top_wgt * 0.24]
        prominent.sort(key=lambda s: (s["x1"] + s["x2"]) / 2.0)

        if len(prominent) >= 3:
            three_plus_cnt += 1

        # CASE 1: All prominent subjects fit 100% inside crop_w together!
        all_l = min(s["x1"] for s in prominent)
        all_r = max(s["x2"] for s in prominent)
        pad_both = max(16, int(crop_w * 0.04))
        if (all_r - all_l + 2 * pad_both) <= crop_w:
            grp_cx = (all_l + all_r) / 2.0
            tx = max(0, min(max_x, int(round(grp_cx - crop_w / 2.0))))
            prev_cx = tx + crop_w / 2.0
            raw_pts.append((t, tx, True, is_cut))
            continue

        # CASE 2: Single Subject
        if len(prominent) == 1:
            p = prominent[0]
            pcx = (p["x1"] + p["x2"]) / 2.0
            # Subtle conversational lead-room in facing direction
            lead_shift = int(p.get("facing", 0) * crop_w * 0.06)
            tx = max(0, min(max_x, int(round(pcx + lead_shift - crop_w / 2.0))))
            tx = _apply_face_border_guard(tx, crop_w, max_x, p, [])
            prev_cx = tx + crop_w / 2.0
            raw_pts.append((t, tx, True, is_cut))
            continue

        # CASE 3: Two Prominent Speakers (or top 2 in wide shot)
        by_speak = sorted(
            prominent,
            key=lambda s: s["weight"] * (s["activity"] ** 1.35) * (1.25 if s["is_face"] else 1.0),
            reverse=True,
        )
        pair = sorted(by_speak[:2], key=lambda s: (s["x1"] + s["x2"]) / 2.0)
        s_left, s_right = pair[0], pair[1]

        score_l = (s_left["weight"] ** 0.55) * (s_left["activity"] ** 1.45)
        score_r = (s_right["weight"] ** 0.55) * (s_right["activity"] ** 1.45)

        if active_side is None or is_cut:
            active_side = "right" if score_r > score_l * 1.05 else "left"
            last_switch_t = t
        else:
            # Hysteresis: switch speaker if new speaker is clearly more active and hold >= 1.6s elapsed
            hold_elapsed = (t - last_switch_t) >= 1.6
            if active_side == "left" and score_r > score_l * 1.22 and (hold_elapsed or score_r > score_l * 1.65):
                active_side = "right"
                last_switch_t = t
            elif active_side == "right" and score_l > score_r * 1.22 and (hold_elapsed or score_l > score_r * 1.65):
                active_side = "left"
                last_switch_t = t

        primary = s_left if active_side == "left" else s_right
        partner = s_right if active_side == "left" else s_left

        outer_span = s_right["x2"] - s_left["x1"]
        outer_pad = max(20, int(crop_w * 0.042))

        if outer_span <= crop_w * 1.34:
            # CONVERSATIONAL GLIMPSE TWO-SHOT:
            # Trim dead background behind Primary so Primary is 100% framed on outer rule-of-thirds
            # AND maximum glimpse (45%-95% of face + hand gestures) of Partner is visible!
            if active_side == "left":
                tx = s_left["x1"] - outer_pad
                # Make sure we reach at least 42% into right partner if possible without cutting left speaker
                min_for_glimpse = int(round(s_right["x1"] + (s_right["x2"] - s_right["x1"]) * 0.48 - crop_w))
                max_allow_left = s_left["x1"] - 12
                if min_for_glimpse <= max_allow_left:
                    tx = max(tx, min_for_glimpse)
            else:
                tx = s_right["x2"] + outer_pad - crop_w
                max_for_glimpse = int(round(s_left["x2"] - (s_left["x2"] - s_left["x1"]) * 0.48))
                min_allow_right = s_right["x2"] + 12 - crop_w
                if max_for_glimpse >= min_allow_right:
                    tx = min(tx, max_for_glimpse)
        else:
            # WIDE TURN-TAKING GIMBAL PAN:
            # Frame active speaker at conversational rule-of-thirds leaning toward listener
            pcx = (primary["x1"] + primary["x2"]) / 2.0
            lean = int(crop_w * 0.11) if active_side == "left" else -int(crop_w * 0.11)
            tx = int(round(pcx + lean - crop_w / 2.0))
            if len(prominent) >= 3:
                unresolvable_cnt += 1

        tx = _apply_face_border_guard(tx, crop_w, max_x, primary, [s for s in prominent if s is not primary])
        prev_cx = tx + crop_w / 2.0
        raw_pts.append((t, tx, True, is_cut))

    keyframes = _smooth_trajectory(raw_pts, default_x, max_x, duration, deadzone_px=34)
    face_coverage = detected_cnt / max(1, len(frames))
    three_plus_ratio = three_plus_cnt / max(1, detected_cnt)
    unresolvable_ratio = unresolvable_cnt / max(1, detected_cnt)

    return {
        "keyframes": keyframes,
        "default_x": default_x,
        "face_coverage": face_coverage,
        "three_plus_ratio": three_plus_ratio,
        "unresolvable_ratio": unresolvable_ratio,
    }


def _analyze_duo_split(analysis: dict) -> dict:
    """
    Evaluates the clip for 9:16 Top/Bottom Duo Split-Screen (`1080x956` + `8px` seam + `1080x956`):
    - Enforces ZERO horizontal overlap (`crop_w <= w // 2` and `tx_r >= tx_l + crop_w`) so a middle
      person can NEVER be duplicated in both top and bottom panels!
    - Tracks dynamic left/right speaker keyframes across scene cuts.
    - Preserves full hats/hair/top-corner overlays via actual `face_y1` headroom calculation.
    """
    w, h = analysis["w"], analysis["h"]
    duration = analysis["duration"]
    frames = analysis["frames"]

    # Waist-up studio podcast framing (~37.5% of source width = 720x638 on 1080p):
    # Gives 480px of horizontal panning freedom to center clustered groups while strictly
    # guaranteeing 0px overlap (tx_r >= tx_l + crop_w) so NO person is ever duplicated!
    panel_aspect = 1080.0 / 956.0
    target_w = int(round(w * 0.375))
    max_half_w = (w // 2) - ((w // 2) % 2)
    crop_w = min(max_half_w, max(240, target_w))
    if crop_w % 2 != 0:
        crop_w -= 1

    crop_h = min(h, int(round(crop_w / panel_aspect)))
    if crop_h % 2 != 0:
        crop_h -= 1
    crop_h = max(240, crop_h)

    max_x = max(0, w - crop_w)
    max_y = max(0, h - crop_h)

    duo_hits = 0
    three_plus_hits = 0
    active_frames = 0
    left_pts = []
    right_pts = []
    top_y_l_samples = []
    top_y_r_samples = []
    lcx_list = []
    rcx_list = []

    default_tx_l = max(0, int(round(w * 0.27 - crop_w / 2.0)))
    default_tx_r = max(default_tx_l + crop_w, min(max_x, int(round(w * 0.73 - crop_w / 2.0))))

    for fd in frames:
        t = fd["t"]
        is_cut = fd.get("scene_cut", False)
        subs = sorted(fd["subjects"], key=lambda s: (s["x1"] + s["x2"]) / 2.0)
        if subs:
            active_frames += 1
        if len(subs) >= 3:
            three_plus_hits += 1

        if len(subs) >= 2:
            top_w = max(s["weight"] for s in subs)
            cand = [s for s in subs if s["weight"] >= top_w * 0.22]
            if len(cand) < 2:
                cand = subs

            outer_sep = ((cand[-1]["x1"] + cand[-1]["x2"]) - (cand[0]["x1"] + cand[0]["x2"])) / 2.0
            if outer_sep >= 0.16 * w:
                duo_hits += 1

                # Find cleanest empty gap between adjacent people near the group midpoint so we NEVER slice a face!
                grp_mid = (cand[0]["x1"] + cand[-1]["x2"]) / 2.0
                best_k = 0
                best_gap_score = -1e9
                for k in range(len(cand) - 1):
                    gap_w = cand[k + 1]["x1"] - cand[k]["x2"]
                    gap_mid = (cand[k]["x2"] + cand[k + 1]["x1"]) / 2.0
                    g_score = gap_w - 0.18 * abs(gap_mid - grp_mid)
                    if g_score > best_gap_score:
                        best_gap_score = g_score
                        best_k = k

                left_grp = cand[: best_k + 1]
                right_grp = cand[best_k + 1 :]
                lcx = (left_grp[0]["x1"] + left_grp[-1]["x2"]) / 2.0
                rcx = (right_grp[0]["x1"] + right_grp[-1]["x2"]) / 2.0
                lcx_list.append(lcx)
                rcx_list.append(rcx)

                split_x = int(round((left_grp[-1]["x2"] + right_grp[0]["x1"]) / 2.0))
                split_x = max(crop_w, min(w - crop_w, split_x))

                want_l = max(0, min(split_x - crop_w, int(round(lcx - crop_w / 2.0))))
                want_r = max(split_x, min(max_x, int(round(rcx - crop_w / 2.0))))

                left_pts.append((t, want_l, True, is_cut))
                right_pts.append((t, want_r, True, is_cut))

                top_l_y1 = min(s["y1"] for s in left_grp)
                top_r_y1 = min(s["y1"] for s in right_grp)
                fh_l = max(30, max(s["y2"] - s["y1"] for s in left_grp))
                fh_r = max(30, max(s["y2"] - s["y1"] for s in right_grp))
                headroom_l = max(72, int(fh_l * 0.78))
                headroom_r = max(72, int(fh_r * 0.78))
                top_y_l_samples.append(max(0, min(max_y, top_l_y1 - headroom_l)))
                top_y_r_samples.append(max(0, min(max_y, top_r_y1 - headroom_r)))
                continue

        left_pts.append((t, default_tx_l, False, is_cut))
        right_pts.append((t, default_tx_r, False, is_cut))

    duo_ratio = duo_hits / max(1, len(frames))
    duo_active_ratio = duo_hits / max(1, active_frames)
    stationary_podcast = False
    if len(lcx_list) >= 4:
        stationary_podcast = (
            np.std(lcx_list) < 0.085 * w
            and np.std(rcx_list) < 0.085 * w
            and (np.median(rcx_list) - np.median(lcx_list)) >= 0.34 * w
            and (three_plus_hits / max(1, active_frames)) <= 0.15
        )

    is_auto_duo = stationary_podcast and (duo_ratio >= 0.48 or duo_active_ratio >= 0.62)

    kfs_l = _smooth_trajectory(left_pts, default_tx_l, max(0, w - 2 * crop_w), duration, deadzone_px=38)
    kfs_r = _smooth_trajectory(right_pts, default_tx_r, max_x, duration, deadzone_px=38)

    # Guarantee non-overlap even after independent trajectory smoothing
    aligned_r = []
    for t_r, xr in kfs_r:
        # Find nearest left x at t_r
        nearest_xl = min(kfs_l, key=lambda item: abs(item[0] - t_r))[1]
        safe_xr = max(xr, min(max_x, nearest_xl + crop_w))
        aligned_r.append((t_r, safe_xr))

    ty_l = int(np.percentile(top_y_l_samples, 25)) if top_y_l_samples else max(0, int(h * 0.04))
    ty_r = int(np.percentile(top_y_r_samples, 25)) if top_y_r_samples else max(0, int(h * 0.04))
    if ty_l % 2 != 0:
        ty_l = max(0, ty_l - 1)
    if ty_r % 2 != 0:
        ty_r = max(0, ty_r - 1)

    return {
        "is_auto_duo": bool(is_auto_duo),
        "has_multi_subjects": duo_hits >= 2 or (duo_active_ratio >= 0.22),
        "crop_w": crop_w,
        "crop_h": crop_h,
        "kfs_l": kfs_l,
        "kfs_r": aligned_r,
        "default_tx_l": default_tx_l,
        "default_tx_r": default_tx_r,
        "ty_l": ty_l,
        "ty_r": ty_r,
    }


def _build_duo_split_filter(analysis: dict, duo_plan: dict) -> str:
    """
    Builds a Zero-Overlap 1080x1920 Duo Split-Screen FFmpeg filtergraph:
    - If >=2 separated subjects exist: Top = Left Speaker(s), Bottom = Right Speaker(s) with 0px overlap!
    - If solo-speaker clip (user forced split_screen on 1 person): Top = Speaker Close-Up,
      Bottom = Wide Full-Context Cinema Stage (never slicing 1 centered person down the nose!).
    """
    w, h = analysis["w"], analysis["h"]
    if not duo_plan["has_multi_subjects"]:
        # Solo-subject graceful Duo Stack: Top Close-Up + Bottom Wide Context
        close_h = min(h, int(h * 0.72))
        if close_h % 2 != 0:
            close_h -= 1
        close_w = min(w, int(round(close_h * (1080.0 / 956.0))))
        if close_w % 2 != 0:
            close_w -= 1
        solo_plan = _plan_face_track_trajectory(analysis, close_w)
        dyn_solo_x = build_crop_expression(solo_plan["keyframes"], solo_plan["default_x"])
        fg_h = min(940, max(400, int(round(1080.0 * h / max(1.0, w))) & ~1))
        pad_top = (956 - fg_h) // 2
        return (
            f"[0:v]split=2[top_in][bot_in];"
            f"[top_in]crop=w={close_w}:h={close_h}:x='{dyn_solo_x}':y=0,"
            f"scale=1080:956:flags=lanczos,pad=1080:964:0:0:color=#111318[top_p];"
            f"[bot_in]split=2[bbg][bfg];"
            f"[bbg]scale=270:240:force_original_aspect_ratio=increase:flags=fast_bilinear,"
            f"crop=270:240,gblur=sigma=14,scale=1080:956:flags=bilinear,eq=brightness=-0.22:saturation=1.15[bbg_o];"
            f"[bfg]scale=1080:{fg_h}:flags=lanczos[bfg_o];"
            f"[bbg_o][bfg_o]overlay=0:{pad_top}:shortest=1[bot_p];"
            f"[top_p][bot_p]vstack=inputs=2,setsar=1[reframed]"
        )

    crop_w = duo_plan["crop_w"]
    crop_h = duo_plan["crop_h"]
    dyn_l = build_crop_expression(duo_plan["kfs_l"], duo_plan["default_tx_l"])
    dyn_r = build_crop_expression(duo_plan["kfs_r"], duo_plan["default_tx_r"])
    ty_l = duo_plan["ty_l"]
    ty_r = duo_plan["ty_r"]

    return (
        f"[0:v]split=2[top_in][bot_in];"
        f"[top_in]crop=w={crop_w}:h={crop_h}:x='{dyn_l}':y={ty_l},"
        f"scale=1080:956:flags=lanczos,"
        f"pad=1080:964:0:0:color=#111318[top_p];"
        f"[bot_in]crop=w={crop_w}:h={crop_h}:x='{dyn_r}':y={ty_r},"
        f"scale=1080:956:flags=lanczos[bot_p];"
        f"[top_p][bot_p]vstack=inputs=2,setsar=1[reframed]"
    )


def _build_smart_fit_9_16_filter(analysis: dict) -> tuple[str, int]:
    """
    Builds a Pro Cinema Vertical Stage (1080x1920) for 3+ person group lineups or wide action:
    - Downscaled Gaussian-blurred ambient backdrop (16x faster!)
    - 4:3 Zoom Stage (1080x810) when group fits inside 72% width, else Full 16:9 Stage (1080x608).
    """
    w, h = analysis["w"], analysis["h"]
    duration = analysis["duration"]
    frames = analysis["frames"]

    spans = []
    for fd in frames:
        subs = fd["subjects"]
        if subs:
            left = min(s["x1"] for s in subs)
            right = max(s["x2"] for s in subs)
            spans.append((right - left) / max(1.0, w))

    p85_span = float(np.percentile(spans, 85)) if spans else 0.85
    use_zoom_stage = (p85_span <= 0.70) and (analysis["side_motion_ratio"] < 0.32) and (w / max(1.0, h) > 1.5)

    bg_chain = (
        "scale=270:480:force_original_aspect_ratio=increase:flags=fast_bilinear,"
        "crop=270:480,"
        "gblur=sigma=16,"
        "scale=1080:1920:flags=bilinear,"
        "eq=brightness=-0.24:saturation=1.18:contrast=1.06"
    )

    if use_zoom_stage:
        stage_crop_w = min(w, int(round(h * 4.0 / 3.0)))
        if stage_crop_w % 2 != 0:
            stage_crop_w -= 1
        max_x = max(0, w - stage_crop_w)
        default_x = max_x // 2

        raw_pts = []
        for fd in frames:
            subs = fd["subjects"]
            if subs:
                cx = (min(s["x1"] for s in subs) + max(s["x2"] for s in subs)) / 2.0
                tx = max(0, min(max_x, int(round(cx - stage_crop_w / 2.0))))
                raw_pts.append((fd["t"], tx, True, fd.get("scene_cut", False)))
            else:
                raw_pts.append((fd["t"], default_x, False, fd.get("scene_cut", False)))

        kfs = _smooth_trajectory(raw_pts, default_x, max_x, duration, deadzone_px=45)
        dyn_x = build_crop_expression(kfs, default_x)
        fg_chain = (
            f"crop=w={stage_crop_w}:h={h}:x='{dyn_x}':y=0,"
            f"scale=1080:810:flags=lanczos,"
            f"pad=1080:818:0:4:color=white@0.18"
        )
        overlay_y = 430
        caption_margin_v = 350
    else:
        fg_h = int(round(1080.0 * h / max(1.0, w)))
        if fg_h % 2 != 0:
            fg_h += 1
        fg_h = min(1400, max(400, fg_h))
        fg_chain = (
            f"scale=1080:{fg_h}:flags=lanczos,"
            f"pad=1080:{fg_h + 8}:0:4:color=white@0.18"
        )
        overlay_y = max(260, (1920 - fg_h) // 2 - 90)
        caption_margin_v = 380

    flt = (
        f"[0:v]split=2[bg_in][fg_in];"
        f"[bg_in]{bg_chain}[bg_out];"
        f"[fg_in]{fg_chain}[fg_out];"
        f"[bg_out][fg_out]overlay=x=0:y={overlay_y}:shortest=1,setsar=1[reframed]"
    )
    return flt, caption_margin_v


def _build_smart_fit_1_1_filter(w: int, h: int) -> tuple[str, int]:
    """Builds a 1080x1080 Square Cinema Stage keeping 100% of wide scenes with blurred square fill."""
    fg_h = int(round(1080.0 * h / max(1.0, w)))
    if fg_h % 2 != 0:
        fg_h += 1
    fg_h = min(1060, max(400, fg_h))
    overlay_y = max(40, (1080 - fg_h) // 2 - 35)
    bg_chain = (
        "scale=270:270:force_original_aspect_ratio=increase:flags=fast_bilinear,"
        "crop=270:270,"
        "gblur=sigma=14,"
        "scale=1080:1080:flags=bilinear,"
        "eq=brightness=-0.24:saturation=1.16"
    )
    fg_chain = f"scale=1080:{fg_h}:flags=lanczos,pad=1080:{fg_h + 6}:0:3:color=white@0.18"
    flt = (
        f"[0:v]split=2[bg_in][fg_in];"
        f"[bg_in]{bg_chain}[bg_out];"
        f"[fg_in]{fg_chain}[fg_out];"
        f"[bg_out][fg_out]overlay=x=0:y={overlay_y}:shortest=1,setsar=1[reframed]"
    )
    return flt, 75


def generate_reframe_plan(
    video_path: str,
    clip_duration: float,
    aspect_ratio: str = "9:16",
    framing_mode: str = "auto",
) -> dict:
    """
    Master AI Reframe Planner primarily engineered for 9:16 Vertical Shorts (plus 1:1 & 16:9):
    - 'auto':
        * Stationary 2-person podcast/interview -> Zero-Overlap Duo Split (`duo_split_9_16`)
        * 1- or 2-person conversation -> YuNet Conversational Glimpse + Active-Speaker Pan (`face_track_9_16`)
        * 3+ person wide lineup or wide landscape action -> Pro Cinema Stage (`cinema_smart_fit_9_16`)
    - 'face_track': YuNet deep face tracking + 2-person Conversational Glimpse + Active-Speaker Panning.
    - 'split_screen': Zero-Overlap (`crop_w <= w/2`, `tx_r >= tx_l + crop_w`) dynamic Duo Split-Screen.
    - 'smart_fit': Full-scene Pro Cinema Canvas with blurred vertical backdrop.
    """
    framing_mode = (framing_mode or "auto").lower().strip()

    # 1. Landscape 16:9 (1920x1080)
    if aspect_ratio == "16:9":
        flt = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1[reframed]"
        )
        return {
            "filter": flt,
            "layout_used": "landscape_16_9",
            "caption_margin_v": 90,
            "caption_alignment": 2,
            "canvas_size": (1920, 1080),
        }

    target_size = (1080, 1920) if aspect_ratio == "9:16" else (1080, 1080)
    analysis = analyze_video_subjects(video_path, clip_duration) if os.path.exists(video_path) else {"valid": False}

    if not analysis.get("valid"):
        if aspect_ratio == "9:16":
            return {
                "filter": "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1[reframed]",
                "layout_used": "fallback_fit",
                "caption_margin_v": 360,
                "caption_alignment": 2,
                "canvas_size": target_size,
            }
        return {
            "filter": "[0:v]crop='min(iw,ih)':'min(iw,ih)':(iw-ow)/2:(ih-oh)/2,scale=1080:1080:flags=lanczos,setsar=1[reframed]",
            "layout_used": "fallback_square",
            "caption_margin_v": 115,
            "caption_alignment": 2,
            "canvas_size": target_size,
        }

    w, h = analysis["w"], analysis["h"]

    # Already native vertical or square source -> direct clean scale
    if aspect_ratio == "9:16" and (w / max(1.0, h)) <= 0.65:
        return {
            "filter": "[0:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1[reframed]",
            "layout_used": "native_vertical",
            "caption_margin_v": 420,
            "caption_alignment": 2,
            "canvas_size": (1080, 1920),
        }
    if aspect_ratio == "1:1" and abs((w / max(1.0, h)) - 1.0) <= 0.08:
        return {
            "filter": "[0:v]scale=1080:1080:flags=lanczos,setsar=1[reframed]",
            "layout_used": "native_square",
            "caption_margin_v": 115,
            "caption_alignment": 2,
            "canvas_size": (1080, 1080),
        }

    # 2. Square 1:1 (1080x1080)
    if aspect_ratio == "1:1":
        if framing_mode == "smart_fit":
            flt, mv = _build_smart_fit_1_1_filter(w, h)
            return {
                "filter": flt,
                "layout_used": "square_smart_fit",
                "caption_margin_v": mv,
                "caption_alignment": 2,
                "canvas_size": (1080, 1080),
            }

        crop_w = min(w, h) & ~1
        sq_plan = _plan_face_track_trajectory(analysis, crop_w)
        if framing_mode == "auto" and (sq_plan["three_plus_ratio"] > 0.35 or analysis["side_motion_ratio"] > 0.45):
            flt, mv = _build_smart_fit_1_1_filter(w, h)
            return {
                "filter": flt,
                "layout_used": "square_smart_fit",
                "caption_margin_v": mv,
                "caption_alignment": 2,
                "canvas_size": (1080, 1080),
            }

        dyn_x = build_crop_expression(sq_plan["keyframes"], sq_plan["default_x"])
        flt = f"[0:v]crop=w={crop_w}:h={crop_w}:x='{dyn_x}':y=0,scale=1080:1080:flags=lanczos,setsar=1[reframed]"
        return {
            "filter": flt,
            "layout_used": "square_face_track",
            "caption_margin_v": 115,
            "caption_alignment": 2,
            "canvas_size": (1080, 1080),
        }

    # 3. Vertical 9:16 (1080x1920)
    duo_plan = _analyze_duo_split(analysis)

    # Explicit or Auto Zero-Overlap Duo Split-Screen
    if framing_mode == "split_screen" or (framing_mode == "auto" and duo_plan["is_auto_duo"]):
        flt = _build_duo_split_filter(analysis, duo_plan)
        return {
            "filter": flt,
            "layout_used": "duo_split_9_16",
            "caption_margin_v": 0,
            "caption_alignment": 5,  # Center seam between top and bottom speaker!
            "canvas_size": (1080, 1920),
        }

    # Explicit Cinema Smart-Fit
    if framing_mode == "smart_fit":
        flt, mv = _build_smart_fit_9_16_filter(analysis)
        return {
            "filter": flt,
            "layout_used": "cinema_smart_fit_9_16",
            "caption_margin_v": mv,
            "caption_alignment": 2,
            "canvas_size": (1080, 1920),
        }

    # Conversational Glimpse + Active-Speaker Gimbal Face-Track (1 or 2 people)
    crop_w = int(round(h * 9.0 / 16.0))
    if crop_w % 2 != 0:
        crop_w += 1
    crop_w = min(w, crop_w)

    track_plan = _plan_face_track_trajectory(analysis, crop_w)

    # In 'auto' mode: use full-bleed 9:16 Face-Track (with 2-shot glimpse & active-speaker pan)
    # for 1- and 2-person clips, and switch to Cinema Smart-Fit ONLY for 3+ person wide lineups
    # or non-human wide action scenes!
    if framing_mode == "auto":
        use_full_vertical_track = (
            track_plan["face_coverage"] >= 0.45
            and track_plan["three_plus_ratio"] <= 0.25
            and track_plan["unresolvable_ratio"] <= 0.20
            and analysis["side_motion_ratio"] <= 0.36
        )
        if not use_full_vertical_track:
            flt, mv = _build_smart_fit_9_16_filter(analysis)
            return {
                "filter": flt,
                "layout_used": "cinema_smart_fit_9_16",
                "caption_margin_v": mv,
                "caption_alignment": 2,
                "canvas_size": (1080, 1920),
            }

    dyn_x = build_crop_expression(track_plan["keyframes"], track_plan["default_x"])
    flt = f"[0:v]crop=w={crop_w}:h={h}:x='{dyn_x}':y=0,scale=1080:1920:flags=lanczos,setsar=1[reframed]"
    return {
        "filter": flt,
        "layout_used": "face_track_9_16",
        "caption_margin_v": 420,
        "caption_alignment": 2,
        "canvas_size": (1080, 1920),
    }


def generate_smart_crop_filter(video_path: str, clip_duration: float, aspect_ratio: str = "9:16") -> str:
    """Backward-compatible helper returning a linear filter if called directly."""
    plan = generate_reframe_plan(video_path, clip_duration, aspect_ratio=aspect_ratio, framing_mode="face_track")
    return plan["filter"].replace("[0:v]", "").replace("[reframed]", "")
