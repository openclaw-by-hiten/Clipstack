import os
import subprocess
from pathlib import Path
from core.audio_ducking import apply_audio_ducking
from core.smart_crop import generate_reframe_plan

_NVENC_AVAILABLE = None


def _has_nvenc() -> bool:
    """Detects once whether hardware-accelerated NVIDIA h264_nvenc is usable (e.g. Google Colab T4 GPU)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        res = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=6,
        )
        _NVENC_AVAILABLE = res.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def escape_ffmpeg_path(path_str: str) -> str:
    """Escapes paths for FFmpeg filter strings (colons and backslashes)."""
    clean_p = path_str.replace("\\", "/")
    if ":" in clean_p:
        parts = clean_p.split(":", 1)
        clean_p = f"{parts[0]}\\:{parts[1]}"
    return clean_p


def _target_dimensions(aspect_ratio: str) -> tuple[int, int]:
    if aspect_ratio == "1:1":
        return 1080, 1080
    if aspect_ratio == "16:9":
        return 1920, 1080
    return 1080, 1920


def render_clip(
    source_video: str,
    clip_start: float,
    clip_duration: float,
    ass_subtitle_path: str,
    output_mp4: str,
    aspect_ratio: str = "9:16",
    framing_mode: str = "auto",
    reframe_plan: dict = None,
    watermark: str = "@clipstack_A1",
    floating_watermark: bool = True,
    header_title: str = None,
    enable_vignette: bool = True,
    enable_grain: bool = False,
    bgm_path: str = None,
    custom_thumbnail_path: str = None,
) -> str:
    """
    Renders the final viral clip:
    - AI Auto-Aligns to 9:16, 1:1, or 16:9 (Face-Track, Duo Split, or Pro Cinema Smart-Fit)
    - Encodes web-streaming-optimized H.264 MP4 (~8-13 MB per 60s instead of 76 MB!) with +faststart
    - Uses NVIDIA NVENC hardware encoding automatically when a GPU (Colab T4) is present
    - Dynamically adapts floating watermark & header coordinates to any canvas size
    """
    if reframe_plan is None:
        reframe_plan = generate_reframe_plan(
            video_path=source_video,
            clip_duration=clip_duration,
            aspect_ratio=aspect_ratio,
            framing_mode=framing_mode,
        )

    reframe_graph = reframe_plan["filter"]
    post_filters = []

    # 1. Crisp mobile detail enhancement
    post_filters.append("unsharp=lx=3:ly=3:la=0.45")

    # 2. Subtle Vignette
    if enable_vignette:
        post_filters.append("vignette=PI/5.2")

    # 3. Compression-safe micro texture (spatial only — avoids 8x temporal noise file-size bloat!)
    if enable_grain:
        post_filters.append("noise=c0s=2:allf=u")

    # Cross-platform font paths
    if os.name == "nt":
        font_impact = "C\\:/Windows/Fonts/impact.ttf"
        font_arial = "C\\:/Windows/Fonts/arial.ttf"
        font_impact_opt = f":fontfile='{font_impact}'"
        font_arial_opt = f":fontfile='{font_arial}'"
    else:
        font_impact = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_arial = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists("/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"):
            font_impact = "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"
        elif os.path.exists("/usr/share/fonts/truetype/msttcorefonts/impact.ttf"):
            font_impact = "/usr/share/fonts/truetype/msttcorefonts/impact.ttf"
        if os.path.exists("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"):
            font_arial = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
        elif os.path.exists("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"):
            font_arial = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
        font_impact_opt = f":fontfile='{font_impact}'" if os.path.exists(font_impact) else ""
        font_arial_opt = f":fontfile='{font_arial}'" if os.path.exists(font_arial) else ""

    # 4. Header / Hook Title Pill Badge
    if header_title:
        clean_title = header_title.replace("'", "").replace(":", "-")[:38]
        esc_title = clean_title.replace(",", "\\,").replace("%", "\\%")
        header_y = 140 if aspect_ratio == "9:16" else 65
        header_fs = 36 if aspect_ratio == "9:16" else 30
        post_filters.append(
            f"drawtext=text='{esc_title}':x=(w-text_w)/2:y={header_y}:"
            f"fontsize={header_fs}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=14:borderw=2:bordercolor=black:"
            f"shadowcolor=black@0.8:shadowx=2:shadowy=2{font_impact_opt}:enable='between(t,0,5)'"
        )

    # 5. Watermark Overlay (Resolution-Independent Dynamic Edge Patrol for 9:16, 1:1, and 16:9!)
    if watermark:
        clean_wm = watermark.replace("'", "").replace(":", "\\:").replace(",", "\\,")
        wm_fs = 32 if aspect_ratio == "9:16" else 26
        top_m = 85 if aspect_ratio == "9:16" else 42
        bot_expr = "(h-text_h-85)" if aspect_ratio == "9:16" else "(h-text_h-42)"
        if floating_watermark:
            x_expr = (
                "if(lt(mod(t\\,14)\\,5.0)\\,"
                "(w-text_w-48)-(w-text_w-96)*(mod(t\\,14)/5.0)\\,"
                "if(lt(mod(t\\,14)\\,7.0)\\,48\\,"
                "if(lt(mod(t\\,14)\\,12.0)\\,48+(w-text_w-96)*((mod(t\\,14)-7.0)/5.0)\\,"
                "w-text_w-48)))"
            )
            y_expr = (
                f"if(lt(mod(t\\,14)\\,5.0)\\,{top_m}\\,"
                f"if(lt(mod(t\\,14)\\,7.0)\\,{top_m}+({bot_expr}-{top_m})*((mod(t\\,14)-5.0)/2.0)\\,"
                f"if(lt(mod(t\\,14)\\,12.0)\\,{bot_expr}\\,"
                f"{bot_expr}-({bot_expr}-{top_m})*((mod(t\\,14)-12.0)/2.0))))"
            )
            post_filters.append(
                f"drawtext=text='{clean_wm}':x='{x_expr}':y='{y_expr}':"
                f"fontsize={wm_fs}:fontcolor=white@0.82{font_arial_opt}:"
                f"shadowcolor=black@0.8:shadowx=2:shadowy=2"
            )
        else:
            post_filters.append(
                f"drawtext=text='{clean_wm}':x=w-text_w-40:y=h-text_h-50:"
                f"fontsize={wm_fs}:fontcolor=white@0.75{font_arial_opt}:shadowcolor=black@0.8:shadowx=2:shadowy=2"
            )

    # 6. Animated Subtitles Burn-in via libass
    if ass_subtitle_path and os.path.exists(ass_subtitle_path):
        esc_ass = escape_ffmpeg_path(ass_subtitle_path)
        post_filters.append(f"subtitles='{esc_ass}'")

    post_filters.append("format=yuv420p")
    full_filter_complex = f"{reframe_graph};[reframed]{','.join(post_filters)}[vout]"

    temp_clip_video = output_mp4.replace(".mp4", "_temp.mp4")

    # Choose GPU NVENC or fast web-optimized libx264 (~1.8-2.4 Mbps -> ~9-13 MB per 60s!)
    use_nvenc = _has_nvenc()
    if use_nvenc:
        v_enc_opts = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "24",
            "-b:v", "2200k",
            "-maxrate", "2800k",
            "-bufsize", "5600k",
            "-profile:v", "high",
        ]
    else:
        v_enc_opts = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-maxrate", "2500k",
            "-bufsize", "5000k",
            "-profile:v", "high",
        ]

    def _build_ffmpeg_cmd(enc_opts: list) -> list:
        return (
            [
                "ffmpeg", "-y",
                "-ss", str(clip_start),
                "-t", str(clip_duration),
                "-i", source_video,
                "-filter_complex", full_filter_complex,
                "-map", "[vout]",
                "-map", "0:a?",
            ]
            + enc_opts
            + [
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-movflags", "+faststart",
                temp_clip_video,
            ]
        )

    try:
        subprocess.run(_build_ffmpeg_cmd(v_enc_opts), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        if use_nvenc:
            # Fallback cleanly to CPU libx264 if NVENC session limit hit
            cpu_opts = [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-maxrate", "2500k",
                "-bufsize", "5000k",
                "-profile:v", "high",
            ]
            subprocess.run(_build_ffmpeg_cmd(cpu_opts), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        else:
            raise

    # 7. Background Music & Audio Ducking (preserving +faststart!)
    if bgm_path and os.path.exists(bgm_path):
        speech_audio = output_mp4.replace(".mp4", "_speech.wav")
        ducked_audio = output_mp4.replace(".mp4", "_ducked.aac")

        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_clip_video, "-vn", speech_audio],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        apply_audio_ducking(speech_audio, bgm_path, ducked_audio)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", temp_clip_video,
                "-i", ducked_audio,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "copy",
                "-movflags", "+faststart",
                output_mp4,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        for p in [temp_clip_video, speech_audio, ducked_audio]:
            if os.path.exists(p):
                os.remove(p)
    else:
        if os.path.exists(output_mp4):
            os.remove(output_mp4)
        os.rename(temp_clip_video, output_mp4)

    # 8. Companion High-Quality Thumbnail matching target aspect ratio
    thumb_path = output_mp4.replace(".mp4", "_thumb.jpg")
    tw, th = _target_dimensions(aspect_ratio)

    if custom_thumbnail_path and os.path.exists(custom_thumbnail_path):
        thumb_cmd = [
            "ffmpeg", "-y",
            "-i", custom_thumbnail_path,
            "-vf", f"scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th}",
            "-q:v", "2",
            thumb_path,
        ]
        subprocess.run(thumb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        seek_start = 3.5 if clip_duration > 7.0 else 0.5
        scan_frames = min(90, int(max(24, (clip_duration - seek_start) * 15)))
        thumb_cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_start),
            "-i", output_mp4,
            "-vf", f"thumbnail=n={scan_frames}",
            "-frames:v", "1",
            "-update", "1",
            "-q:v", "2",
            thumb_path,
        ]
        try:
            subprocess.run(thumb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "2.0", "-i", output_mp4, "-frames:v", "1", "-update", "1", "-q:v", "2", thumb_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

    try:
        from core.branding import composite_brand_on_thumbnail
        composite_brand_on_thumbnail(thumb_path)
    except Exception as e:
        print(f"[Renderer] Branding compositing error: {e}")

    return output_mp4
