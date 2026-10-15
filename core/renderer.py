import os
import subprocess
from pathlib import Path
from core.audio_ducking import apply_audio_ducking

def escape_ffmpeg_path(path_str: str) -> str:
    """Escapes Windows paths for FFmpeg filter strings (colons and backslashes)."""
    clean_p = path_str.replace("\\", "/")
    if ":" in clean_p:
        parts = clean_p.split(":", 1)
        clean_p = f"{parts[0]}\\:{parts[1]}"
    return clean_p

def render_clip(
    source_video: str,
    clip_start: float,
    clip_duration: float,
    ass_subtitle_path: str,
    output_mp4: str,
    aspect_ratio: str = "9:16",
    watermark: str = "@clipstack_A1",
    floating_watermark: bool = True,
    header_title: str = None,
    enable_vignette: bool = True,
    enable_grain: bool = True,
    bgm_path: str = None,
    custom_thumbnail_path: str = None
) -> str:
    """
    Renders the final viral clip:
    - Crops to 9:16, 1:1, or 16:9
    - Applies visual atmosphere filters (vignette, film grain, unsharp)
    - Burns animated .ass subtitles
    - Overlays branding watermark and top header text with explicit Windows fonts
    - Mixes and ducks background audio (if provided)
    - Generates companion thumbnail
    """
    filters = []

    # 1. Aspect Ratio Reframing
    if aspect_ratio == "9:16":
        filters.append("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)':(iw-ow)/2:(ih-oh)/2,scale=1080:1920:flags=lanczos")
    elif aspect_ratio == "1:1":
        filters.append("crop='min(iw,ih)':'min(iw,ih)':(iw-ow)/2:(ih-oh)/2,scale=1080:1080:flags=lanczos")
    else:  # 16:9
        filters.append("scale=1920:1080:flags=lanczos")

    # 2. Sharpening filter for crisp mobile display
    filters.append("unsharp=lx=3:ly=3:la=0.6")

    # 3. Atmosphere: Vignette
    if enable_vignette:
        filters.append("vignette=PI/4.5")

    # 4. Atmosphere: Subtle Film Grain
    if enable_grain:
        filters.append("noise=alls=8:allf=t+u")

    # Cross-platform font path detection for Windows & Linux / Colab
    if os.name == "nt":
        font_impact = "C\\:/Windows/Fonts/impact.ttf"
        font_arial = "C\\:/Windows/Fonts/arial.ttf"
        font_impact_opt = f":fontfile='{font_impact}'"
        font_arial_opt = f":fontfile='{font_arial}'"
    else:
        # Linux / Colab font paths
        font_impact = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_arial = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists("/usr/share/fonts/truetype/msttcorefonts/impact.ttf"):
            font_impact = "/usr/share/fonts/truetype/msttcorefonts/impact.ttf"
        if os.path.exists("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"):
            font_arial = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
        font_impact_opt = f":fontfile='{font_impact}'" if os.path.exists(font_impact) else ""
        font_arial_opt = f":fontfile='{font_arial}'" if os.path.exists(font_arial) else ""

    # 5. Header / Title Text (Sleek modern pill badge with opening hook visibility)
    if header_title:
        clean_title = header_title.replace("'", "").replace(":", "-")[:35]
        # In FFmpeg drawtext filter, escape commas and percent signs
        esc_title = clean_title.replace(",", "\\,").replace("%", "\\%")
        filters.append(
            f"drawtext=text='{esc_title}':x=(w-text_w)/2:y=140:"
            f"fontsize=38:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=14:borderw=2:bordercolor=black:"
            f"shadowcolor=black@0.8:shadowx=2:shadowy=2{font_impact_opt}:enable='between(t,0,5)'"
        )

    # 6. Watermark Overlay (Anti-AI Dynamic Floating or Static)
    if watermark:
        clean_wm = watermark.replace("'", "")
        if floating_watermark:
            # Perimeter Edge Patrol (Anti-Crop & Anti-AI Inpainting):
            # 1. Appears top-right and glides horizontally to top-left (y=100)
            # 2. Glides down along the left margin to bottom-left
            # 3. Traverses horizontally across the bottom margin to bottom-right (y=1700)
            # 4. Glides up along the right margin to top-right, and repeats
            # Leaves the center 80% viewing angle completely unobstructed!
            x_expr = (
                "if(lt(mod(t\\,12)\\,4.5)\\,"
                "(w-text_w-60)-(w-text_w-120)*(mod(t\\,12)/4.5)\\,"
                "if(lt(mod(t\\,12)\\,6.0)\\,60\\,"
                "if(lt(mod(t\\,12)\\,10.5)\\,60+(w-text_w-120)*((mod(t\\,12)-6.0)/4.5)\\,"
                "w-text_w-60)))"
            )
            y_expr = (
                "if(lt(mod(t\\,12)\\,4.5)\\,100\\,"
                "if(lt(mod(t\\,12)\\,6.0)\\,100+(1700-100)*((mod(t\\,12)-4.5)/1.5)\\,"
                "if(lt(mod(t\\,12)\\,10.5)\\,1700\\,"
                "1700-(1700-100)*((mod(t\\,12)-10.5)/1.5))))"
            )
            filters.append(
                f"drawtext=text='{clean_wm}':x='{x_expr}':y='{y_expr}':"
                f"fontsize=34:fontcolor=white@0.82{font_arial_opt}:"
                f"shadowcolor=black@0.8:shadowx=2:shadowy=2"
            )
        else:
            filters.append(
                f"drawtext=text='{clean_wm}':x=w-text_w-40:y=h-text_h-60:"
                f"fontsize=34:fontcolor=white@0.75{font_arial_opt}:shadowcolor=black@0.8:shadowx=2:shadowy=2"
            )

    # 7. Animated Subtitles Burn-in via libass
    if ass_subtitle_path and os.path.exists(ass_subtitle_path):
        esc_ass = escape_ffmpeg_path(ass_subtitle_path)
        filters.append(f"subtitles='{esc_ass}'")

    filter_complex = ",".join(filters)

    # Render video slice
    temp_clip_video = output_mp4.replace(".mp4", "_temp.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-t", str(clip_duration),
        "-i", source_video,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        temp_clip_video
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 8. Background Music & Audio Ducking
    if bgm_path and os.path.exists(bgm_path):
        speech_audio = output_mp4.replace(".mp4", "_speech.wav")
        ducked_audio = output_mp4.replace(".mp4", "_ducked.aac")
        
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_clip_video, "-vn", speech_audio],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
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
                output_mp4
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        for p in [temp_clip_video, speech_audio, ducked_audio]:
            if os.path.exists(p):
                os.remove(p)
    else:
        if os.path.exists(output_mp4):
            os.remove(output_mp4)
        os.rename(temp_clip_video, output_mp4)

    # 9. Generate Companion High-Quality Smart Thumbnail for Clip Library
    thumb_path = output_mp4.replace(".mp4", "_thumb.jpg")
    
    if custom_thumbnail_path and os.path.exists(custom_thumbnail_path):
        # Professional AI Thumbnail Integration
        thumb_cmd = [
            "ffmpeg", "-y",
            "-i", custom_thumbnail_path,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-q:v", "2",
            thumb_path
        ]
        subprocess.run(thumb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        seek_start = 3.5 if clip_duration > 7.0 else 0.5
        scan_frames = min(120, int(max(30, (clip_duration - seek_start) * 20)))

        thumb_cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_start),
            "-i", output_mp4,
            "-vf", f"thumbnail=n={scan_frames}",
            "-frames:v", "1",
            "-update", "1",
            "-q:v", "2",
            thumb_path
        ]
        try:
            subprocess.run(thumb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            # Fallback to direct frame capture if thumbnail filter fails
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "3.0", "-i", output_mp4, "-frames:v", "1", "-update", "1", "-q:v", "2", thumb_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )

    # 10. Composite active channel logo badge onto thumbnail
    try:
        from core.branding import composite_brand_on_thumbnail
        composite_brand_on_thumbnail(thumb_path)
    except Exception as e:
        print(f"[Renderer] Branding compositing error: {e}")

    return output_mp4
