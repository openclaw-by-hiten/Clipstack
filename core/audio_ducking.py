import os
import subprocess

def apply_audio_ducking(
    speech_audio_path: str,
    bgm_path: str,
    output_audio_path: str,
    bgm_volume: float = 0.28,
    ducking_ratio: float = 6.0
) -> str:
    """
    Applies FFmpeg sidechain compression to automatically duck background music
    whenever voice/dialogue is active.
    """
    if not os.path.exists(bgm_path):
        return speech_audio_path

    # sidechaincompress filter graph:
    # 1. Background music volume reduced to base background level
    # 2. Sidechain compressor uses speech track to duck BGM
    # 3. amix combines ducked BGM with full clear speech
    filter_graph = (
        f"[1:a]volume={bgm_volume}[bgm];"
        f"[bgm][0:a]sidechaincompress=threshold=0.04:ratio={ducking_ratio}:attack=80:release=700[ducked_bgm];"
        f"[0:a][ducked_bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", speech_audio_path,
        "-i", bgm_path,
        "-filter_complex", filter_graph,
        "-map", "[aout]",
        "-c:a", "aac",
        "-b:a", "192k",
        output_audio_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_audio_path
