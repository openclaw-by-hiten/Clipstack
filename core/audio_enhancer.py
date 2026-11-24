import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

_DEEPFILTER_STATE = None


def _try_demucs_vocals(wav_in: str, work_dir: Path) -> Optional[str]:
    """Optional Demucs vocal isolation (strips copyrighted background music / loud game BGM)."""
    exe = shutil.which("demucs")
    if not exe:
        return None
    try:
        out_root = work_dir / "demucs_out"
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            exe,
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-o", str(out_root),
            wav_in,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=180)
        stem_name = Path(wav_in).stem
        vocals_wav = out_root / "htdemucs" / stem_name / "vocals.wav"
        if vocals_wav.exists() and vocals_wav.stat().st_size > 10_000:
            print("[AudioEnhancer] Demucs isolated clean vocals (background music stripped).")
            return str(vocals_wav)
    except Exception as e:
        print(f"[AudioEnhancer] Demucs fallback notice: {e}")
    return None


def _try_deepfilternet(wav_in: str, wav_out: str) -> bool:
    """Neural real-time speech cleanup via DeepFilterNet3 if installed."""
    global _DEEPFILTER_STATE
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except Exception:
        return False

    try:
        if _DEEPFILTER_STATE is None:
            model, df_state, _ = init_df()
            _DEEPFILTER_STATE = (model, df_state)
        model, df_state = _DEEPFILTER_STATE
        audio, _ = load_audio(wav_in, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(wav_out, enhanced, df_state.sr())
        if os.path.exists(wav_out) and os.path.getsize(wav_out) > 10_000:
            print("[AudioEnhancer] DeepFilterNet3 studio voice denoising applied.")
            return True
    except Exception as e:
        print(f"[AudioEnhancer] DeepFilterNet fallback notice: {e}")
    return False


def enhance_clip_audio(
    source_video: str,
    output_audio_aac: str,
    studio_voice: bool = True,
    strip_music: bool = False,
) -> Optional[str]:
    """
    Studio audio mastering pipeline for viral short-form platforms (-14 LUFS target):
    1. Extracts full-fidelity 48kHz WAV from the rendered/sliced clip.
    2. Optional Demucs vocal isolation when strip_music=True (avoids copyright music claims).
    3. Optional DeepFilterNet3 neural noise suppression (falls back to FFmpeg spectral denoise).
    4. Broadcast mastering chain: highpass rumble cut + gentle vocal presence compression +
       EBU R128 loudnorm (-14 LUFS, -1.5 dBTP) encoded to 192k AAC.
    """
    if not source_video or not os.path.exists(source_video):
        return None

    out_p = Path(output_audio_aac)
    work_dir = out_p.parent
    raw_wav = str(out_p.with_suffix("")) + "_raw48k.wav"
    df_wav = str(out_p.with_suffix("")) + "_df48k.wav"
    temp_files = [raw_wav, df_wav]

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", source_video,
                "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
                raw_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=45,
        )
        if not os.path.exists(raw_wav) or os.path.getsize(raw_wav) < 4000:
            return None

        active_wav = raw_wav

        if strip_music:
            demucs_wav = _try_demucs_vocals(active_wav, work_dir)
            if demucs_wav:
                active_wav = demucs_wav

        used_neural_df = False
        if studio_voice:
            if _try_deepfilternet(active_wav, df_wav):
                active_wav = df_wav
                used_neural_df = True

        # Broadcast mastering chain: clean low-end rumble + vocal punch + -14 LUFS loudness norm
        af_chain = []
        if studio_voice:
            af_chain.append("highpass=f=70")
            if not used_neural_df:
                # Light spectral noise reduction that preserves game SFX while cutting mic hiss
                af_chain.append("afftdn=nf=-28:tn=1")
            af_chain.append("acompressor=threshold=-18dB:ratio=2.4:attack=12:release=140:makeup=2")
        af_chain.append("loudnorm=I=-14:TP=-1.5:LRA=11")

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", active_wav,
                "-af", ",".join(af_chain),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                output_audio_aac,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        )
        if os.path.exists(output_audio_aac) and os.path.getsize(output_audio_aac) > 4000:
            return output_audio_aac
    except Exception as e:
        print(f"[AudioEnhancer] Studio audio fallback (keeping original audio): {e}")
    finally:
        for tp in temp_files:
            if os.path.exists(tp):
                try:
                    os.remove(tp)
                except OSError:
                    pass
        demucs_dir = work_dir / "demucs_out"
        if demucs_dir.exists():
            shutil.rmtree(demucs_dir, ignore_errors=True)
    return None
