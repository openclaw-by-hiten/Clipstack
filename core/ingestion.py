import os
import subprocess
import yt_dlp
from pathlib import Path
from core.config import WORKSPACE_DIR

NODE_EXE = r"C:\Program Files\nodejs\node.exe"

def get_base_ydl_opts() -> dict:
    """Returns resilient base options configuring Node.js JS challenge solver."""
    opts = {
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
    }
    if os.path.exists(NODE_EXE):
        opts['js_runtimes'] = {'node': {'path': NODE_EXE}}
        opts['remote_components'] = ['ejs:github']
    return opts

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

    ydl_opts = get_base_ydl_opts()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
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
    using HTTP Range Requests with Node challenge solver (Zero full-video download!).
    """
    if progress_callback:
        progress_callback(f"Streaming on-demand video slice [{int(start_sec)}s - {int(end_sec)}s] from YouTube...")

    ydl_opts = get_base_ydl_opts()
    ydl_opts.update({
        'format': 'bv*[height<=1080]+ba/b',
        'outtmpl': output_path,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
        'force_keyframes_at_cuts': True
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
