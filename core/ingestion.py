import os
import time
import shutil
import subprocess
import yt_dlp
from pathlib import Path
from core.config import WORKSPACE_DIR

# Cross-platform Node.js binary detection for JS challenge solving (Windows & Colab / Linux)
if os.name == "nt" and os.path.exists(r"C:\Program Files\nodejs\node.exe"):
    NODE_BIN = r"C:\Program Files\nodejs\node.exe"
else:
    NODE_BIN = shutil.which("node")

# Session cache for YouTube metadata & stream URLs (prevents duplicate 3-second network extractions)
_YOUTUBE_INFO_CACHE = {}

def get_base_ydl_opts() -> dict:
    """Returns resilient base options configuring Node.js JS challenge solver if available."""
    opts = {
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
    }
    if NODE_BIN and os.path.exists(NODE_BIN):
        opts['js_runtimes'] = {'node': {'path': NODE_BIN}}
        opts['remote_components'] = ['ejs:github']
    return opts

def get_youtube_info(url: str) -> dict:
    """Fetches and caches YouTube video metadata and direct streaming URLs for 1 hour."""
    now = time.time()
    if url in _YOUTUBE_INFO_CACHE:
        entry_time, data = _YOUTUBE_INFO_CACHE[url]
        if now - entry_time < 3600:
            return data

    ydl_opts = get_base_ydl_opts()
    ydl_opts['format'] = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best'
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    _YOUTUBE_INFO_CACHE[url] = (now, info)
    return info

def get_media_duration(file_path: str) -> float:
    """Uses ffprobe to return the exact duration of a media file in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"Error checking duration with ffprobe: {e}")
        return 0.0

def download_youtube_audio_only(url: str, progress_callback=None) -> dict:
    """
    STEP 1: Downloads ONLY the lightweight audio stream (takes ~3-5 seconds, ~20-30MB).
    Bypasses YouTube 403 Forbidden by using Node JS challenge solver and universal audio format.
    """
    if progress_callback:
        progress_callback("Connecting to YouTube: Fetching metadata & audio stream only...")

    info = get_youtube_info(url)
    video_id = info.get("id", "yt_video")
    title = info.get("title", "YouTube Video")
    duration = float(info.get("duration", 0.0))

    raw_audio_template = str(WORKSPACE_DIR / f"{video_id}_temp_audio.%(ext)s")
    target_wav_path = str(WORKSPACE_DIR / f"{video_id}_audio.wav")

    def yt_hook(d):
        if d['status'] == 'downloading' and progress_callback:
            percent = d.get('_percent_str', '0%').replace('\x1b[0;94m', '').replace('\x1b[0m', '')
            progress_callback(f"Downloading lightweight audio: {percent}")

    download_opts = get_base_ydl_opts()
    download_opts.update({
        'format': 'ba/bestaudio/best',  # Universal audio (opus/webm/m4a) without 403 restrictions
        'outtmpl': raw_audio_template,
        'progress_hooks': [yt_hook]
    })

    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download([url])

    # Find the downloaded raw audio file
    downloaded_raw = None
    for p in WORKSPACE_DIR.glob(f"{video_id}_temp_audio.*"):
        downloaded_raw = str(p)
        break

    if not downloaded_raw or not os.path.exists(downloaded_raw):
        raise RuntimeError("Audio download failed. Check network or YouTube URL.")

    # Convert to standard 16kHz mono WAV for Whisper transcription & acoustic analysis
    cmd_audio = [
        "ffmpeg", "-y",
        "-i", downloaded_raw,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        target_wav_path
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Cleanup temp raw audio container
    if os.path.exists(downloaded_raw):
        try:
            os.remove(downloaded_raw)
        except Exception:
            pass

    return {
        "id": video_id,
        "title": title,
        "audio_path": target_wav_path,
        "duration": duration,
        "url": url
    }

def download_youtube_video_section(url: str, start_sec: float, end_sec: float, output_path: str, progress_callback=None) -> str:
    """
    STEP 3: Downloads ONLY the specific 30s-180s section of video from YouTube
    in ~5-15 seconds using high-speed HTTP byte-range seek via FFmpeg (Zero full-video download!).
    """
    duration = max(1.0, end_sec - start_sec)
    if progress_callback:
        progress_callback(f"Streaming on-demand video slice [{int(start_sec)}s - {int(end_sec)}s] from YouTube...")

    try:
        info = get_youtube_info(url)
        formats = info.get("requested_formats") or [info]
        video_url = formats[0].get("url")
        audio_url = formats[1].get("url") if len(formats) > 1 else video_url

        if video_url:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", video_url,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", audio_url,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 50000:
                return output_path
    except Exception as e:
        print(f"[StreamSeek] Fast seek error ({e}). Trying fallback range download...")

    # Fallback to direct yt-dlp section download without the slow force_keyframes_at_cuts
    ydl_opts = get_base_ydl_opts()
    ydl_opts.update({
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        'outtmpl': output_path,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
    })

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path

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
        audio_path
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    duration = get_media_duration(file_path)

    return {
        "id": video_id,
        "title": path_obj.stem,
        "video_path": file_path,
        "audio_path": audio_path,
        "duration": duration,
        "url": None
    }
