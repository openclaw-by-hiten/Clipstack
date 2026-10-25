import os
import cv2
import numpy as np


def _get_cascades():
    """Loads OpenCV Haar cascades for frontal face, profile face, and upper body."""
    base = cv2.data.haarcascades
    frontal_p = os.path.join(base, "haarcascade_frontalface_default.xml")
    profile_p = os.path.join(base, "haarcascade_profileface.xml")
    upper_p = os.path.join(base, "haarcascade_upperbody.xml")

    frontal = cv2.CascadeClassifier(frontal_p) if os.path.exists(frontal_p) else None
    profile = cv2.CascadeClassifier(profile_p) if os.path.exists(profile_p) else None
    upper = cv2.CascadeClassifier(upper_p) if os.path.exists(upper_p) else None
    return frontal, profile, upper


def _merge_boxes(boxes: list) -> list:
    """Merges horizontally/spatially overlapping subject boxes: (x1, y1, x2, y2, weight, is_face)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for bx in boxes:
        x1, y1, x2, y2, wgt, is_face = bx
        merged = False
        for idx, kb in enumerate(kept):
            kx1, ky1, kx2, ky2, kwgt, kis_face = kb
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = max(1, (x2 - x1) * (y2 - y1))
            area_b = max(1, (kx2 - kx1) * (ky2 - ky1))
            # Also merge if centers are horizontally very close (e.g. face inside upper body)
            cx_a, cx_b = (x1 + x2) / 2.0, (kx1 + kx2) / 2.0
            if inter / min(area_a, area_b) > 0.35 or abs(cx_a - cx_b) < min(x2 - x1, kx2 - kx1) * 0.55:
                # Keep face box tighter if one is face, else union
                if kis_face and not is_face:
                    merged = True
                    break
                if is_face and not kis_face:
                    kept[idx] = bx
                    merged = True
                    break
                kept[idx] = (
                    min(x1, kx1),
                    min(y1, ky1),
                    max(x2, kx2),
                    max(y2, ky2),
                    max(wgt, kwgt),
                    kis_face or is_face,
                )
                merged = True
                break
        if not merged:
            kept.append(bx)
    return kept


def build_crop_expression(keyframes: list, default_x: int) -> str:
    """Builds a smooth piecewise-linear FFmpeg crop x expression from (t, x) keyframes."""
    if not keyframes or len(keyframes) <= 1:
        return str(keyframes[0][1] if keyframes else default_x)

    # Cap keyframe count to prevent deep FFmpeg expression parser recursion
    if len(keyframes) > 22:
        step = max(1, len(keyframes) // 20)
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


def _smooth_trajectory(raw_points: list, default_x: int, max_x: int, duration: float, deadzone_px: int = 38) -> list:
    """
    Applies forward/backward fill, camera deadzone hysteresis (prevents micro-jitter),
    and EMA smoothing to produce clean cinematographic camera keyframes.
    raw_points: list of (t, target_x, detected_bool)
    """
    if not raw_points:
        return [(0.0, default_x), (duration, default_x)]

    # Backward fill initial undetected frames from first real detection
    first_valid = next((x for _, x, det in raw_points if det), default_x)
    filled = []
    last_x = first_valid
    for t, x, det in raw_points:
        if det:
            last_x = x
        filled.append((t, last_x))

    # Deadzone + EMA smoothing
    smoothed = []
    cam_x = float(filled[0][1])
    anchor_x = cam_x
    for t, target_x in filled:
        if abs(target_x - anchor_x) > deadzone_px:
            anchor_x = float(target_x)
        cam_x = 0.35 * anchor_x + 0.65 * cam_x
        clamped = max(0, min(max_x, int(round(cam_x))))
        smoothed.append((t, clamped))

    keyframes = [smoothed[0]]
    for t, x in smoothed[1:]:
        last_t, last_kx = keyframes[-1]
        if abs(x - last_kx) >= 24 or (t - last_t) >= 2.5:
            keyframes.append((t, x))

    if keyframes[-1][0] < duration:
        keyframes.append((duration, smoothed[-1][1]))
    return keyframes


def analyze_video_subjects(video_path: str, clip_duration: float) -> dict:
    """
    Samples frames across the clip to measure:
    - Video resolution (w, h) & actual duration
    - Per-frame face & upper-body bounding boxes (never blindly averaging distant people!)
    - Horizontal edge-activity ratio (detects wide action/graphics near left/right margins)
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

    frontal, profile, upper = _get_cascades()
    num_samples = min(32, max(6, int(duration / 1.8)))
    scale = min(1.0, 640.0 / max(1, w))
    inv_scale = 1.0 / scale

    frames_data = []
    prev_small_gray = None
    side_motion_hits = 0
    motion_samples = 0

    for i in range(num_samples):
        t = min(duration - 0.05, (i + 0.3) * (duration / num_samples))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sh, sw = gray.shape[:2]

        # Measure horizontal motion spread (left 24% & right 24% vs center 52%)
        if prev_small_gray is not None and prev_small_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_small_gray)
            left_e = float(np.mean(diff[:, : int(sw * 0.24)]))
            right_e = float(np.mean(diff[:, int(sw * 0.76) :]))
            mid_e = float(np.mean(diff[:, int(sw * 0.24) : int(sw * 0.76)])) + 1e-4
            motion_samples += 1
            if (left_e > 6.5 and right_e > 6.5) or ((left_e + right_e) / 2.0 > mid_e * 0.85 and (left_e + right_e) > 11.0):
                side_motion_hits += 1
        prev_small_gray = gray

        raw_boxes = []
        min_face = max(20, int(sh * 0.075))
        if frontal is not None:
            det_f = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face))
            for (fx, fy, fw, fh) in det_f:
                x1, y1 = int(fx * inv_scale), int(fy * inv_scale)
                x2, y2 = int((fx + fw) * inv_scale), int((fy + fh) * inv_scale)
                raw_boxes.append((x1, y1, x2, y2, float(fw * fh * 2.0), True))

        if profile is not None and len(raw_boxes) < 2:
            det_p = profile.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face))
            for (fx, fy, fw, fh) in det_p:
                x1, y1 = int(fx * inv_scale), int(fy * inv_scale)
                x2, y2 = int((fx + fw) * inv_scale), int((fy + fh) * inv_scale)
                raw_boxes.append((x1, y1, x2, y2, float(fw * fh * 1.6), True))
            # Flipped profile catches faces looking the opposite direction
            gray_flip = cv2.flip(gray, 1)
            det_pf = profile.detectMultiScale(gray_flip, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face))
            for (fx, fy, fw, fh) in det_pf:
                orig_fx = sw - fx - fw
                x1, y1 = int(orig_fx * inv_scale), int(fy * inv_scale)
                x2, y2 = int((orig_fx + fw) * inv_scale), int((fy + fh) * inv_scale)
                raw_boxes.append((x1, y1, x2, y2, float(fw * fh * 1.6), True))

        if upper is not None and len(raw_boxes) == 0:
            min_ub = max(36, int(sh * 0.16))
            det_u = upper.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(min_ub, min_ub))
            for (ux, uy, uw, uh) in det_u:
                x1, y1 = int(ux * inv_scale), int(uy * inv_scale)
                x2, y2 = int((ux + uw) * inv_scale), int((uy + uh) * inv_scale)
                raw_boxes.append((x1, y1, x2, y2, float(uw * uh * 0.8), False))

        subjects = _merge_boxes(raw_boxes)
        frames_data.append({"t": t, "subjects": subjects})

    cap.release()
    side_motion_ratio = side_motion_hits / max(1, motion_samples)
    return {
        "valid": True,
        "w": w,
        "h": h,
        "duration": duration,
        "frames": frames_data,
        "side_motion_ratio": side_motion_ratio,
    }


def _choose_primary_subject_x(subjects: list, crop_w: int, w: int, prev_center_x: float) -> tuple[int, bool, bool]:
    """
    Picks a crop_x that centers the primary speaker WITHOUT averaging distant faces!
    Also checks if ANY other detected subject would be sliced in half by [crop_x, crop_x + crop_w].
    Returns (crop_x, detected, has_edge_collision).
    """
    max_x = max(0, w - crop_w)
    default_x = max_x // 2
    if not subjects:
        return default_x, False, False

    # Sort subjects by priority: face + size + proximity to previous tracked speaker
    scored = []
    for s in subjects:
        x1, y1, x2, y2, wgt, is_face = s
        cx = (x1 + x2) / 2.0
        continuity_bonus = max(0.6, 1.25 - (abs(cx - prev_center_x) / max(1.0, w)))
        scored.append((wgt * continuity_bonus * (1.4 if is_face else 1.0), s))
    scored.sort(key=lambda item: item[0], reverse=True)
    primary = scored[0][1]

    px1, py1, px2, py2, _, _ = primary
    pcx = (px1 + px2) / 2.0
    crop_x = max(0, min(max_x, int(round(pcx - crop_w / 2.0))))

    # Check if all subjects actually fit inside ONE crop_w window together!
    all_left = min(s[0] for s in subjects)
    all_right = max(s[2] for s in subjects)
    pad = int(crop_w * 0.08)
    if (all_right - all_left + 2 * pad) <= crop_w:
        group_cx = (all_left + all_right) / 2.0
        crop_x = max(0, min(max_x, int(round(group_cx - crop_w / 2.0))))
        return crop_x, True, False

    # Check if any secondary subject is sliced in half or another prominent speaker is excluded
    has_collision = False
    crop_r = crop_x + crop_w
    for s in subjects:
        if s is primary:
            # Check extreme close-up forehead/side cut
            if (px2 - px1) > crop_w * 0.78 or py1 < 12:
                has_collision = True
            continue
        sx1, sy1, sx2, sy2, swgt, sis_face = s
        # If secondary subject is at least 28% as prominent as primary:
        if swgt >= primary[4] * 0.28:
            # Either sliced across crop border OR completely outside in a 2-person shot!
            sliced_left = sx1 < crop_x < sx2
            sliced_right = sx1 < crop_r < sx2
            outside = sx2 <= crop_x or sx1 >= crop_r
            if sliced_left or sliced_right or (sis_face and outside):
                has_collision = True
                break

    return crop_x, True, has_collision


def _check_duo_split_candidate(analysis: dict) -> dict:
    """
    Checks if the clip is a clean 2-person Left/Right conversation suitable for
    9:16 Top/Bottom Duo Split-Screen (`1080x960` + `1080x960`).
    """
    w, h = analysis["w"], analysis["h"]
    frames = analysis["frames"]
    duo_pairs = []
    for fd in frames:
        faces = [s for s in fd["subjects"] if s[5]]
        if len(faces) == 2:
            f_sorted = sorted(faces, key=lambda b: (b[0] + b[2]) / 2.0)
            lcx = (f_sorted[0][0] + f_sorted[0][2]) / 2.0
            lcy = (f_sorted[0][1] + f_sorted[0][3]) / 2.0
            rcx = (f_sorted[1][0] + f_sorted[1][2]) / 2.0
            rcy = (f_sorted[1][1] + f_sorted[1][3]) / 2.0
            if lcx < 0.47 * w and rcx > 0.53 * w and (rcx - lcx) >= 0.26 * w:
                duo_pairs.append((lcx, lcy, rcx, rcy))

    active_frames = max(1, sum(1 for fd in frames if fd["subjects"]))
    duo_ratio = len(duo_pairs) / max(1, len(frames))
    duo_active_ratio = len(duo_pairs) / active_frames

    if len(duo_pairs) >= 3 and (duo_ratio >= 0.40 or duo_active_ratio >= 0.55):
        arr = np.array(duo_pairs)
        # Ensure reasonably stationary speakers (podcast/interview/two-shot)
        if np.std(arr[:, 0]) < 0.12 * w and np.std(arr[:, 2]) < 0.12 * w:
            return {
                "is_duo": True,
                "lx": float(np.median(arr[:, 0])),
                "ly": float(np.median(arr[:, 1])),
                "rx": float(np.median(arr[:, 2])),
                "ry": float(np.median(arr[:, 3])),
            }
    return {"is_duo": False}


def _build_duo_split_filter(w: int, h: int, lx: float, ly: float, rx: float, ry: float) -> str:
    """Builds a 1080x1920 Top/Bottom Split-Screen FFmpeg filtergraph centering Speaker L and Speaker R."""
    # Each panel is 1080 x 958 (aspect 1.1273), stacked with a 4px divider = 1080 x 1920
    panel_aspect = 1080.0 / 958.0
    crop_h = min(h, int(h * 0.88))
    if crop_h % 2 != 0:
        crop_h -= 1
    crop_w = min(w, int(round(crop_h * panel_aspect)))
    if crop_w % 2 != 0:
        crop_w -= 1

    max_x = max(0, w - crop_w)
    max_y = max(0, h - crop_h)

    # Frame head slightly in upper-middle of each panel
    tx_l = max(0, min(max_x, int(round(lx - crop_w / 2.0))))
    ty_l = max(0, min(max_y, int(round(ly - crop_h * 0.38))))

    tx_r = max(0, min(max_x, int(round(rx - crop_w / 2.0))))
    ty_r = max(0, min(max_y, int(round(ry - crop_h * 0.38))))

    return (
        f"[0:v]split=2[top_in][bot_in];"
        f"[top_in]crop={crop_w}:{crop_h}:{tx_l}:{ty_l},scale=1080:958:flags=lanczos,"
        f"pad=1080:962:0:0:color=#18181b[top_p];"
        f"[bot_in]crop={crop_w}:{crop_h}:{tx_r}:{ty_r},scale=1080:958:flags=lanczos[bot_p];"
        f"[top_p][bot_p]vstack=inputs=2,setsar=1[reframed]"
    )


def _build_smart_fit_9_16_filter(analysis: dict) -> tuple[str, int]:
    """
    Builds a Pro Cinema Vertical Stage (1080x1920):
    - Ultra-fast Gaussian-blurred ambient backdrop (downscaled blur -> 16x faster!)
    - Crisp foreground stage:
      * If subjects fit within 74% width: 4:3 Cinema Stage (1080x810) smoothly panned to keep everyone big & 100% visible!
      * If subjects span edge-to-edge (>74% width): Full 16:9 Stage (1080x608) so ZERO people or text are ever cut off on the sides!
    - Returns (filter_str, safe_caption_margin_v).
    """
    w, h = analysis["w"], analysis["h"]
    duration = analysis["duration"]
    frames = analysis["frames"]

    spans = []
    for fd in frames:
        subs = fd["subjects"]
        if subs:
            left = min(s[0] for s in subs)
            right = max(s[2] for s in subs)
            spans.append((right - left) / max(1.0, w))

    p85_span = float(np.percentile(spans, 85)) if spans else 0.85
    use_zoom_stage = (p85_span <= 0.68) and (analysis["side_motion_ratio"] < 0.30) and (w / max(1.0, h) > 1.5)

    bg_chain = (
        "scale=270:480:force_original_aspect_ratio=increase:flags=fast_bilinear,"
        "crop=270:480,"
        "gblur=sigma=16,"
        "scale=1080:1920:flags=bilinear,"
        "eq=brightness=-0.24:saturation=1.18:contrast=1.06"
    )

    if use_zoom_stage:
        # 4:3 Cinema Stage (keeps 75% of horizontal 16:9 width -> 1080x810 stage)
        stage_crop_w = min(w, int(round(h * 4.0 / 3.0)))
        if stage_crop_w % 2 != 0:
            stage_crop_w -= 1
        max_x = max(0, w - stage_crop_w)
        default_x = max_x // 2

        raw_pts = []
        for fd in frames:
            subs = fd["subjects"]
            if subs:
                cx = (min(s[0] for s in subs) + max(s[2] for s in subs)) / 2.0
                tx = max(0, min(max_x, int(round(cx - stage_crop_w / 2.0))))
                raw_pts.append((fd["t"], tx, True))
            else:
                raw_pts.append((fd["t"], default_x, False))

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
        # Full 100% Width Cinema Stage (1080x608) -> ZERO side cropping of any person or graphic!
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
    Master AI Reframe Planner supporting 9:16, 1:1, and 16:9 with 4 modes:
    - 'auto': Intelligently picks Face-Track, Duo Split-Screen, or Pro Cinema Smart-Fit
              so NO person is EVER sliced in half on the left/right edges!
    - 'smart_fit': Full-scene Pro Cinema Canvas with blurred vertical backdrop (0% side crop).
    - 'face_track': Smooth gimbal speaker tracking (locks onto primary speaker, never averages 2 faces).
    - 'split_screen': Top/Bottom 2-speaker podcast stack (falls back to smart_fit if <2 faces).

    Returns:
        {
            "filter": str (FFmpeg filtergraph taking [0:v] and producing [reframed]),
            "layout_used": str,
            "caption_margin_v": int,
            "caption_alignment": int (2=bottom-center, 5=middle-center),
            "canvas_size": (int, int)
        }
    """
    framing_mode = (framing_mode or "auto").lower().strip()

    # 1. Landscape 16:9 (1920x1080) — Preserve 100% of frame cleanly
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
    duration = analysis["duration"]
    frames = analysis["frames"]

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

        crop_w = min(w, h)
        if crop_w % 2 != 0:
            crop_w -= 1
        max_x = max(0, w - crop_w)
        default_x = max_x // 2

        raw_pts = []
        collisions = 0
        detected_cnt = 0
        prev_cx = w / 2.0
        for fd in frames:
            tx, det, col = _choose_primary_subject_x(fd["subjects"], crop_w, w, prev_cx)
            if det:
                detected_cnt += 1
                prev_cx = tx + crop_w / 2.0
            if col:
                collisions += 1
            raw_pts.append((fd["t"], tx, det))

        collision_ratio = collisions / max(1, detected_cnt) if detected_cnt > 0 else 0.0
        if framing_mode == "auto" and (collision_ratio > 0.35 or analysis["side_motion_ratio"] > 0.45):
            flt, mv = _build_smart_fit_1_1_filter(w, h)
            return {
                "filter": flt,
                "layout_used": "square_smart_fit",
                "caption_margin_v": mv,
                "caption_alignment": 2,
                "canvas_size": (1080, 1080),
            }

        kfs = _smooth_trajectory(raw_pts, default_x, max_x, duration)
        dyn_x = build_crop_expression(kfs, default_x)
        flt = f"[0:v]crop=w={crop_w}:h={crop_w}:x='{dyn_x}':y=0,scale=1080:1080:flags=lanczos,setsar=1[reframed]"
        return {
            "filter": flt,
            "layout_used": "square_face_track",
            "caption_margin_v": 115,
            "caption_alignment": 2,
            "canvas_size": (1080, 1080),
        }

    # 3. Vertical 9:16 (1080x1920)
    duo_info = _check_duo_split_candidate(analysis)

    # Explicit or Auto Duo Split-Screen
    if framing_mode == "split_screen" or (framing_mode == "auto" and duo_info["is_duo"]):
        if duo_info["is_duo"]:
            flt = _build_duo_split_filter(w, h, duo_info["lx"], duo_info["ly"], duo_info["rx"], duo_info["ry"])
            return {
                "filter": flt,
                "layout_used": "duo_split_9_16",
                "caption_margin_v": 0,
                "caption_alignment": 5,  # Center seam between top and bottom speaker!
                "canvas_size": (1080, 1920),
            }
        elif framing_mode == "split_screen":
            # User explicitly requested split_screen even without 2 auto-detected faces -> left third + right third
            flt = _build_duo_split_filter(w, h, w * 0.30, h * 0.42, w * 0.70, h * 0.42)
            return {
                "filter": flt,
                "layout_used": "duo_split_9_16",
                "caption_margin_v": 0,
                "caption_alignment": 5,
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

    # Evaluate Single-Speaker Face-Track viability
    crop_w = int(round(h * 9.0 / 16.0))
    if crop_w % 2 != 0:
        crop_w += 1
    crop_w = min(w, crop_w)
    max_x = max(0, w - crop_w)
    default_x = max_x // 2

    raw_pts = []
    collisions = 0
    detected_cnt = 0
    prev_cx = w / 2.0
    for fd in frames:
        tx, det, col = _choose_primary_subject_x(fd["subjects"], crop_w, w, prev_cx)
        if det:
            detected_cnt += 1
            prev_cx = tx + crop_w / 2.0
        if col:
            collisions += 1
        raw_pts.append((fd["t"], tx, det))

    face_coverage = detected_cnt / max(1, len(frames))
    collision_ratio = collisions / max(1, detected_cnt) if detected_cnt > 0 else 1.0

    # In 'auto' mode: only use full 9:16 zoom-crop if a single speaker is cleanly tracked
    # with minimal edge collisions and low wide side-motion; otherwise use Pro Cinema Smart-Fit
    # so NOBODY is ever cut in half!
    if framing_mode == "auto":
        clean_solo_speaker = (
            face_coverage >= 0.55
            and collision_ratio <= 0.18
            and analysis["side_motion_ratio"] <= 0.22
        )
        if not clean_solo_speaker:
            flt, mv = _build_smart_fit_9_16_filter(analysis)
            return {
                "filter": flt,
                "layout_used": "cinema_smart_fit_9_16",
                "caption_margin_v": mv,
                "caption_alignment": 2,
                "canvas_size": (1080, 1920),
            }

    kfs = _smooth_trajectory(raw_pts, default_x, max_x, duration)
    dyn_x = build_crop_expression(kfs, default_x)
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
