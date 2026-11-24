import asyncio
import os
import subprocess
from pathlib import Path
from typing import Optional

_CHATTERBOX_MODEL = None


def _try_chatterbox_tts(text: str, out_wav: str) -> bool:
    """Optional Resemble AI Chatterbox expressive TTS if installed on CUDA GPU."""
    global _CHATTERBOX_MODEL
    try:
        import torch
        import torchaudio
        if not torch.cuda.is_available():
            return False
        from chatterbox.tts import ChatterboxTTS
    except Exception:
        return False

    try:
        if _CHATTERBOX_MODEL is None:
            print("[HookVoiceover] Loading Chatterbox TTS on CUDA GPU...")
            _CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(device="cuda")
        wav = _CHATTERBOX_MODEL.generate(text)
        torchaudio.save(out_wav, wav, _CHATTERBOX_MODEL.sr)
        torch.cuda.empty_cache()
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 4000:
            print("[HookVoiceover] Generated expressive Chatterbox viral hook voiceover.")
            return True
    except Exception as e:
        print(f"[HookVoiceover] Chatterbox fallback notice: {e}")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return False


def _try_edge_tts(text: str, out_wav: str) -> bool:
    """Zero-GPU neural voice fallback via edge-tts (if installed)."""
    try:
        import edge_tts
    except Exception:
        return False

    mp3_tmp = str(Path(out_wav).with_suffix(".mp3"))
    try:
        async def _synth():
            comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+8%")
            await comm.save(mp3_tmp)

        asyncio.run(_synth())
        if os.path.exists(mp3_tmp) and os.path.getsize(mp3_tmp) > 2000:
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_tmp, "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", out_wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=20,
            )
            return os.path.exists(out_wav) and os.path.getsize(out_wav) > 4000
    except Exception as e:
        print(f"[HookVoiceover] edge-tts fallback notice: {e}")
    finally:
        if os.path.exists(mp3_tmp):
            try:
                os.remove(mp3_tmp)
            except OSError:
                pass
    return False


def generate_hook_voiceover(hook_text: str, output_wav: str) -> Optional[str]:
    """
    Generates a crisp 2-4 second spoken viral hook intro WAV from a single line of text.
    Priority: Chatterbox TTS (CUDA GPU) -> edge-tts (zero-GPU neural voice).
    Returns path to WAV or None if unavailable.
    """
    clean = " ".join((hook_text or "").strip().split())
    if len(clean) < 4:
        return None
    clean = clean[:160]

    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    if _try_chatterbox_tts(clean, output_wav):
        return output_wav
    if _try_edge_tts(clean, output_wav):
        return output_wav
    return None
