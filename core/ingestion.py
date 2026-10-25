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
    """
    Returns resilient base options bypassing YouTube bot detection and Error 152.
    Emulates InnerTube Android mobile client which bypasses web bot challenges on cloud/datacenter IPs.
    """
    opts = {
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/130.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Sec-Fetch-Mode': 'navigate',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android'],
                'player_skip': ['webpage', 'configs', 'js']
            }
        }
    }

    # Automatically load cookies.txt if provided by user or saved in workspace
    cookie_candidates = [
        WORKSPACE_DIR / "cookies.txt",
        Path(__file__).resolve().parent.parent / "cookies.txt",
        Path.cwd() / "cookies.txt",
        Path("/content/cookies.txt"),
        Path("/content/Clipstack/cookies.txt"),
        Path("/content/Clipstack/workspace/cookies.txt"),
    ]
    for cp in cookie_candidates:
        if cp.exists() and cp.stat().st_size > 0:
            opts['cookiefile'] = str(cp)
            break

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
    ydl_opts['format'] = 'ba/b/best'
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
    Bypasses YouTube bot check and Error 152 by streaming directly from CDN stream URL via FFmpeg.
    """
    if progress_callback:
        progress_callback("Connecting to YouTube: Fetching metadata & audio stream only...")

    info = get_youtube_info(url)
    video_id = info.get("id", "yt_video")
    title = info.get("title", "YouTube Video")
    duration = float(info.get("duration", 0.0))

    target_wav_path = str(WORKSPACE_DIR / f"{video_id}_audio.wav")

    # If already cached and valid, reuse immediately
    if os.path.exists(target_wav_path) and os.path.getsize(target_wav_path) > 100000:
        return {
            "id": video_id,
            "title": title,
            "audio_path": target_wav_path,
            "duration": duration,
            "url": url
        }

    # Extract audio directly from stream URL via FFmpeg (Zero second API request / Zero bot check)
    formats = info.get("requested_formats") or info.get("formats") or [info]
    stream_url = None
    for f in reversed(formats):
        if f.get("url") and f.get("acodec") != "none":
            stream_url = f.get("url")
            break

    if not stream_url:
        for f in formats:
            if f.get("url"):
                stream_url = f.get("url")
                break

    if stream_url:
        if progress_callback:
            progress_callback("Streaming audio directly from CDN via high-speed FFmpeg...")
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", stream_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            target_wav_path
        ]
        try:
            subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if os.path.exists(target_wav_path) and os.path.getsize(target_wav_path) > 100000:
                return {
                    "id": video_id,
                    "title": title,
                    "audio_path": target_wav_path,
                    "duration": duration,
                    "url": url
                }
        except Exception as e:
            print(f"[DirectAudioStream] FFmpeg stream extraction error: {e}. Trying fallback...")

    # Fallback with yt-dlp
    raw_audio_template = str(WORKSPACE_DIR / f"{video_id}_temp_audio.%(ext)s")
    def yt_hook(d):
        if d['status'] == 'downloading' and progress_callback:
            percent = d.get('_percent_str', '0%').replace('\x1b[0;94m', '').replace('\x1b[0m', '')
            progress_callback(f"Downloading lightweight audio: {percent}")

    download_opts = get_base_ydl_opts()
    download_opts.update({
        'format': 'ba/b/bestaudio/best',
        'outtmpl': raw_audio_template,
        'progress_hooks': [yt_hook]
    })

    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"Audio download failed: {e}")

    downloaded_raw = None
    for p in WORKSPACE_DIR.glob(f"{video_id}_temp_audio.*"):
        downloaded_raw = str(p)
        break

    if not downloaded_raw or not os.path.exists(downloaded_raw):
        raise RuntimeError("Audio download failed. Check network or YouTube URL.")

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
    in ~3-8 seconds using high-speed HTTP byte-range seek via FFmpeg (Zero full-video download!).
    """
    duration = max(1.0, end_sec - start_sec)
    if progress_callback:
        progress_callback(f"Streaming on-demand video slice [{int(start_sec)}s - {int(end_sec)}s] from YouTube...")

    try:
        info = get_youtube_info(url)
        formats = info.get("requested_formats") or info.get("formats") or [info]
        video_url = None
        audio_url = None

        for f in formats:
            if f.get("url"):
                if f.get("vcodec") != "none" and not video_url:
                    video_url = f.get("url")
                if f.get("acodec") != "none" and not audio_url:
                    audio_url = f.get("url")

        if not video_url:
            video_url = formats[0].get("url")
        if not audio_url:
            audio_url = video_url

        if video_url:
            if video_url == audio_url:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_sec),
                    "-t", str(duration),
                    "-i", video_url,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "22",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
            else:
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
                    str(output_path)
                ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 50000:
                return output_path
    except Exception as e:
        print(f"[StreamSeek] Fast seek error ({e}). Trying fallback range download...")

    # Fallback to direct yt-dlp section download
    ydl_opts = get_base_ydl_opts()
    ydl_opts.update({
        'format': 'ba/b/best[height<=720]/best',
        'outtmpl': str(output_path),
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
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
