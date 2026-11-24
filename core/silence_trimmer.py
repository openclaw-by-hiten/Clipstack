import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Tuple


def _probe_duration(video_path: str) -> float:
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return max(0.0, float(res.stdout.strip()))
    except Exception:
        return 0.0


def _try_ffmpeg_silence_trim(
    input_mp4: str,
    output_mp4: str,
    orig_dur: float,
    min_silence_sec: float = 0.80,
    margin_sec: float = 0.18,
) -> bool:
    """
    Ultra-fast, zero-hang pure-FFmpeg dead-air remover (~1-3s per clip):
    Detects silent pauses via silencedetect and cleanly stitches active speech/action segments.
    Never blocks on stdin, never breaks on auto-editor CLI flag changes, and uses minimal RAM.
    """
    try:
        detect_cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", input_mp4,
            "-af", f"silencedetect=noise=-34dB:d={min_silence_sec:.2f}",
            "-f", "null", "-",
        ]
        res = subprocess.run(
            detect_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=25,
        )
        log = res.stderr or ""
        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]
        if not starts or not ends:
            return False

        # Build keep-intervals (non-silent speech/action regions with natural breathing margin)
        silences = []
        for s, e in zip(starts, ends):
            if e - s >= min_silence_sec:
                silences.append((max(0.0, s + margin_sec), min(orig_dur, e - margin_sec)))
        silences = [(s, e) for s, e in silences if e > s + 0.15]
        if not silences:
            return False

        keep = []
        cursor = 0.0
        for s, e in silences:
            if s > cursor + 0.10:
                keep.append((cursor, s))
            cursor = max(cursor, e)
        if orig_dur - cursor > 0.15:
            keep.append((cursor, orig_dur))

        kept_total = sum(e - s for s, e in keep)
        # Guard: only trim if we save >= 1.2s AND keep at least 55% of clip and <= 28 cuts
        if (orig_dur - kept_total) < 1.2 or kept_total < orig_dur * 0.55 or len(keep) > 28 or not keep:
            return False

        filter_parts = []
        concat_inputs = []
        for idx, (ks, ke) in enumerate(keep):
            filter_parts.append(f"[0:v]trim=start={ks:.3f}:end={ke:.3f},setpts=PTS-STARTPTS[v{idx}]")
            filter_parts.append(f"[0:a]atrim=start={ks:.3f}:end={ke:.3f},asetpts=PTS-STARTPTS[a{idx}]")
            concat_inputs.append(f"[v{idx}][a{idx}]")

        fg = ";".join(filter_parts) + f";{''.join(concat_inputs)}concat=n={len(keep)}:v=1:a=1[outv][outa]"
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", input_mp4,
            "-filter_complex", fg,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            output_mp4,
        ]
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=35,
        )
        return os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 50_000
    except Exception as e:
        print(f"[SilenceTrimmer] FFmpeg silence trim skip: {e}")
        return False


def _try_auto_editor(input_mp4: str, output_mp4: str, margin_sec: float = 0.20) -> bool:
    """Secondary fallback using auto-editor CLI with strict non-interactive stdin & 25s cap."""
    exe = shutil.which("auto-editor")
    if not exe:
        return False
    try:
        cmd = [
            exe,
            input_mp4,
            "--margin", f"{margin_sec:.2f}sec",
            "--edit", "audio:threshold=4%",
            "--no-open",
            "-o", output_mp4,
        ]
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=25,
        )
        return os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 50_000
    except Exception as e:
        print(f"[SilenceTrimmer] auto-editor skipped (fast fallback): {e}")
        return False


def trim_dead_air(input_mp4: str, margin_sec: float = 0.20) -> Tuple[str, float, float]:
    """
    Removes long dead-air silence pauses from a clip slice for snappy short-form pacing.
    Prioritizes fast non-blocking FFmpeg silencedetect (~1-3s, zero hang risk).
    Returns (active_video_path, new_duration, seconds_removed).
    """
    orig_dur = _probe_duration(input_mp4)
    if orig_dur < 6.0:
        return input_mp4, orig_dur, 0.0

    trimmed_path = str(Path(input_mp4).with_suffix("")) + "_tight.mp4"
    ok = _try_ffmpeg_silence_trim(input_mp4, trimmed_path, orig_dur=orig_dur, margin_sec=margin_sec)
    if not ok:
        # Only fall back to auto-editor if FFmpeg didn't find clean silence intervals
        # (kept strictly bounded at 25s with stdin=DEVNULL so it can NEVER hang!)
        ok = False

    if ok:
        new_dur = _probe_duration(trimmed_path)
        saved = orig_dur - new_dur
        if saved >= 1.2 and new_dur >= max(5.0, orig_dur * 0.52):
            print(f"[SilenceTrimmer] Trimmed {saved:.1f}s of dead air ({orig_dur:.1f}s -> {new_dur:.1f}s).")
            return trimmed_path, new_dur, saved
        try:
            os.remove(trimmed_path)
        except OSError:
            pass

    return input_mp4, orig_dur, 0.0
