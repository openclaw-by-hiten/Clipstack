import os
import sys

# Permanently block broken/conflicting yt-dlp plugins (such as yt-dlp-youtube-oauth2)
os.environ["YTDLP_NO_PLUGINS"] = "1"

import time
import shutil
import subprocess
from pathlib import Path

# Auto-purge yt-dlp-youtube-oauth2 if cached in the Python environment
try:
    import importlib.util
    if importlib.util.find_spec("yt_dlp_plugins"):
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "yt-dlp-youtube-oauth2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
except Exception:
    pass

import yt_dlp
from core.config import WORKSPACE_DIR
from core.warp_shield import (
    ensure_warp_proxy,
    rotate_warp_proxy,
    get_js_runtime_opts,
    get_proxy_env,
)

# Session cache for YouTube metadata & stream URLs (prevents duplicate network extractions)
_YOUTUBE_INFO_CACHE = {}


def get_base_ydl_opts(use_warp: bool = True) -> dict:
    """
    Returns resilient, zero-cookie yt-dlp options backed by userspace Cloudflare WARP Shield
    and automatic Node.js/Deno JS challenge solving.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "nocheckcertificate": True,
    }

    if use_warp:
        proxy_url = ensure_warp_proxy()
        if proxy_url:
            opts["proxy"] = proxy_url

    opts.update(get_js_runtime_opts())
    return opts


def _is_bot_or_rate_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "not a bot",
            "sign in",
            "429",
            "too many requests",
            "unable to connect to proxy",
            "proxyerror",
        )
    )


def get_youtube_info(url: str, force_refresh: bool = False) -> dict:
    """
    Fetches and caches YouTube metadata + HLS/HTTPS stream URLs.
    Automatically activates Cloudflare WARP Shield and rotates IP if challenged.
    """
    now = time.time()
    if not force_refresh and url in _YOUTUBE_INFO_CACHE:
        entry_time, data = _YOUTUBE_INFO_CACHE[url]
        if now - entry_time < 1800:
            return data

    fmt_selector = (
        "bestvideo[protocol^=m3u8][vcodec^=avc1][height<=1080]+bestaudio/"
        "bestvideo[protocol^=m3u8][height<=1080]+bestaudio/"
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
    )

    last_err = None
    # Attempt 1: Active WARP Shield; Attempt 2: Rotate WARP identity for fresh IP; Attempt 3: Direct fallback
    for attempt in range(3):
        if attempt == 1:
            rotate_warp_proxy()
        use_warp = attempt < 2
        ydl_opts = get_base_ydl_opts(use_warp=use_warp)
        ydl_opts["format"] = fmt_selector

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            info["_active_proxy"] = ydl_opts.get("proxy")
            _YOUTUBE_INFO_CACHE[url] = (time.time(), info)
            return info
        except Exception as e:
            last_err = e
            print(f"[Ingestion] YouTube metadata attempt {attempt + 1} failed: {e}")
            if not _is_bot_or_rate_error(e) and attempt == 0:
                continue

    raise RuntimeError(f"Could not extract video metadata from YouTube: {last_err}")


def get_media_duration(file_path: str) -> float:
    """Uses ffprobe to return the exact duration of a media file in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"Error checking duration with ffprobe: {e}")
        return 0.0


def download_youtube_audio_only(url: str, progress_callback=None) -> dict:
    """
    STEP 1: Downloads ONLY lightweight speech-grade audio (~12-20MB) via Cloudflare WARP Shield
    with multi-fragment concurrency and converts to 16kHz mono WAV for Whisper & acoustic analysis.
    """
    if progress_callback:
        progress_callback("Connecting via Cloudflare WARP Shield: Fetching metadata & audio stream...")

    info = get_youtube_info(url)
    video_id = info.get("id", "yt_video")
    title = info.get("title", "YouTube Video")
    duration = float(info.get("duration", 0.0))
    active_proxy = info.get("_active_proxy")

    target_wav_path = str(WORKSPACE_DIR / f"{video_id}_audio.wav")

    # Reuse valid cached WAV if already extracted
    if os.path.exists(target_wav_path) and os.path.getsize(target_wav_path) > 100_000:
        return {
            "id": video_id,
            "title": title,
            "audio_path": target_wav_path,
            "duration": duration,
            "url": url,
        }

    # Cleanup any partial temp files from prior runs
    for stale in WORKSPACE_DIR.glob(f"{video_id}_temp_audio.*"):
        try:
            stale.unlink()
        except OSError:
            pass

    raw_audio_template = str(WORKSPACE_DIR / f"{video_id}_temp_audio.%(ext)s")

    def yt_hook(d):
        if d.get("status") == "downloading" and progress_callback:
            percent = d.get("_percent_str", "0%").replace("\x1b[0;94m", "").replace("\x1b[0m", "").strip()
            progress_callback(f"Downloading lightweight audio stream: {percent}")

    download_opts = get_base_ydl_opts(use_warp=bool(active_proxy))
    if active_proxy:
        download_opts["proxy"] = active_proxy
    download_opts.update({
        "format": "ba[protocol=https][abr<=75]/ba[abr<=75]/ba[ext=m4a]/ba/bestaudio/best",
        "outtmpl": raw_audio_template,
        "concurrent_fragment_downloads": 8,
        "progress_hooks": [yt_hook],
    })

    downloaded_raw = None
    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])
        for p in WORKSPACE_DIR.glob(f"{video_id}_temp_audio.*"):
            if p.suffix != ".part" and p.stat().st_size > 10_000:
                downloaded_raw = str(p)
                break
    except Exception as e:
        print(f"[Ingestion] Fast yt-dlp audio download error ({e}), trying direct FFmpeg stream fallback...")

    if downloaded_raw and os.path.exists(downloaded_raw):
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", downloaded_raw,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            target_wav_path,
        ]
        subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        try:
            os.remove(downloaded_raw)
        except OSError:
            pass
    else:
        # Fallback: Direct FFmpeg stream over WARP proxy
        formats = info.get("requested_formats") or info.get("formats") or [info]
        stream_url = None
        for f in formats:
            if f.get("url") and f.get("acodec") not in (None, "none"):
                stream_url = f.get("url")
                break
        if not stream_url:
            raise RuntimeError("Audio download failed: No playable audio stream URL returned.")

        cmd_audio = [
            "ffmpeg", "-y",
            "-i", stream_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            target_wav_path,
        ]
        subprocess.run(
            cmd_audio,
            env=get_proxy_env(active_proxy),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )

    if not os.path.exists(target_wav_path) or os.path.getsize(target_wav_path) < 50_000:
        raise RuntimeError("Audio extraction produced an empty WAV file.")

    return {
        "id": video_id,
        "title": title,
        "audio_path": target_wav_path,
        "duration": duration,
        "url": url,
    }


def _select_fast_video_stream(info: dict) -> tuple[str, bool]:
    """
    Selects the fastest unthrottled HD video stream URL from cached yt-dlp info.
    Prefers 1080p H.264 (`avc1`) HLS `.m3u8` (ratebypass `.ts` segments, ~15x faster than raw HTTPS).
    Returns (video_url, can_stream_copy_h264).
    """
    all_fmts = info.get("formats") or []

    # 1. Best: 1080p/720p H.264 (avc1) over HLS m3u8 (allows -c:v copy and unthrottled .ts chunks)
    hls_avc = [
        f for f in all_fmts
        if f.get("url")
        and (f.get("protocol") or "").startswith("m3u8")
        and (f.get("vcodec") or "").startswith("avc1")
        and 480 <= (f.get("height") or 0) <= 1080
    ]
    if hls_avc:
        hls_avc.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
        return hls_avc[-1]["url"], True

    # 2. Next best: Any HLS m3u8 video <= 1080p
    hls_any = [
        f for f in all_fmts
        if f.get("url")
        and (f.get("protocol") or "").startswith("m3u8")
        and f.get("vcodec") not in (None, "none")
        and 480 <= (f.get("height") or 0) <= 1080
    ]
    if hls_any:
        hls_any.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
        best_f = hls_any[-1]
        is_avc = (best_f.get("vcodec") or "").startswith("avc1")
        return best_f["url"], is_avc

    # 3. Fallback: requested_formats[0]
    req_fmts = info.get("requested_formats") or []
    if req_fmts and req_fmts[0].get("url"):
        is_avc = (req_fmts[0].get("vcodec") or "").startswith("avc1")
        return req_fmts[0]["url"], is_avc

    return None, False


def download_youtube_video_section(url: str, start_sec: float, end_sec: float, output_path: str, progress_callback=None) -> str:
    """
    STEP 3: Downloads ONLY the specific 30s-180s section of HD video from YouTube
    using unthrottled HLS `.ts` segment seeking over Cloudflare WARP Shield + local cached WAV audio.
    """
    duration = max(1.0, end_sec - start_sec)
    if progress_callback:
        progress_callback(f"Streaming on-demand HD video slice [{int(start_sec)}s - {int(end_sec)}s]...")

    active_proxy = None
    try:
        info = get_youtube_info(url)
        active_proxy = info.get("_active_proxy")
        video_id = info.get("id", "yt_video")
        local_wav = WORKSPACE_DIR / f"{video_id}_audio.wav"

        video_url, is_avc = _select_fast_video_stream(info)
        if video_url:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", video_url,
            ]
            if local_wav.exists() and local_wav.stat().st_size > 50_000:
                cmd.extend([
                    "-ss", str(start_sec),
                    "-t", str(duration),
                    "-i", str(local_wav),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                ])

            if is_avc:
                cmd.extend(["-c:v", "copy"])
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"])

            cmd.extend([
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path),
            ])

            subprocess.run(
                cmd,
                env=get_proxy_env(active_proxy),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=180,
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 50_000:
                return str(output_path)
    except Exception as e:
        print(f"[StreamSeek] Fast HLS slice error ({e}). Trying yt-dlp range fallback...")

    # Fallback: yt-dlp HLS range download over WARP Shield
    ydl_opts = get_base_ydl_opts(use_warp=True)
    if active_proxy:
        ydl_opts["proxy"] = active_proxy
    ydl_opts.update({
        "format": (
            "bv*[protocol^=m3u8][vcodec^=avc1][height<=1080]+ba[protocol^=m3u8]/"
            "bv*[protocol^=m3u8][height<=720]+ba[protocol^=m3u8]/"
            "best[height<=720]/best"
        ),
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "concurrent_fragment_downloads": 8,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
    })

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return str(output_path)


def process_local_file(file_path: str, progress_callback=None) -> dict:
    """Prepares an uploaded local video file for processing."""
    if progress_callback:
        progress_callback("Validating uploaded local video file...")

    path_obj = Path(file_path)
    video_id = path_obj.stem
    audio_path = str(WORKSPACE_DIR / f"{video_id}_audio.wav")

    cmd_audio = [
        "ffmpeg", "-y",
        "-i", file_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    duration = get_media_duration(file_path)

    return {
        "id": video_id,
        "title": path_obj.stem,
        "video_path": file_path,
        "audio_path": audio_path,
        "duration": duration,
        "url": None,
    }
