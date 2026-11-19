import os
import subprocess
import uuid
from typing import List, Dict
from core.config import WORKSPACE_DIR, MAX_CLIPS_PER_SESSION
from core.ingestion import download_youtube_video_section

def is_overlapping(cand_start: float, cand_end: float, accepted_clips: List[Dict], buffer_sec: float = 8.0) -> bool:
    """Ensures that approved clips have at least buffer_sec space between them."""
    for clip in accepted_clips:
        c_start = clip["start"] - buffer_sec
        c_end = clip["end"] + buffer_sec
        if not (cand_end <= c_start or cand_start >= c_end):
            return True
    return False

def verify_and_select_clips_streaming(
    scorer,
    yt_url: str,
    local_video_path: str,
    raw_audio_path: str,
    candidates: List[Dict],
    target_count: int,
    target_duration: int,
    progress_callback=None
) -> List[Dict]:
    """
    STREAMING VERIFICATION LOOP:
    1. Tests candidates in ranked order of viral energy.
    2. Uses local video if cached (cuts in 0.05s) or on-demand stream slice.
    3. Feeds slice to Meta TRIBE v2 / Neural Quality Gate.
    4. If score < threshold: Discards slice and repeats with the next best candidate.
    5. If score >= threshold: Approves clip and keeps video slice for final rendering.
    """
    target_count = min(target_count, MAX_CLIPS_PER_SESSION)
    accepted = []
    rejected_pool = []
    total_candidates = len(candidates)
    
    threshold_map = {30: 60.0, 60: 58.0, 90: 55.0, 180: 52.0}
    current_threshold = threshold_map.get(target_duration, 58.0)

    # Check if full video was previously downloaded or cached in workspace
    cached_video = local_video_path
    if not cached_video and yt_url:
        clean_id = yt_url.split("v=")[-1].split("&")[0].split("/")[-1].replace("?", "")
        for p in WORKSPACE_DIR.glob(f"source_{clean_id}.*"):
            if p.suffix in [".mp4", ".mkv", ".webm"] and p.stat().st_size > 10 * 1024 * 1024:
                cached_video = str(p)
                print(f"[CACHE] Using existing local high-speed source: {p.name}")
                break

    tested_count = 0
    run_uid = uuid.uuid4().hex[:8]

    if progress_callback:
        progress_callback(f"Starting On-Demand Neural Verification (Target: {target_count} clips of {target_duration}s)...")

    for i, cand in enumerate(candidates):
        if len(accepted) >= target_count:
            break

        tested_count += 1
        # Prevent getting stuck on flat audio
        if tested_count > 6 and len(accepted) == 0:
            current_threshold = max(55.0, current_threshold - 5.0)

        # Check overlap with already accepted clips
        if is_overlapping(cand["start"], cand["end"], accepted):
            continue

        cand_slice_video = str(WORKSPACE_DIR / f"temp_slice_{run_uid}_{tested_count}.mp4")
        cand_slice_audio = str(WORKSPACE_DIR / f"temp_slice_{run_uid}_{tested_count}.wav")

        if progress_callback:
            progress_callback(
                f"Candidate #{tested_count} [{int(cand['start'])}s - {int(cand['end'])}s] (Viral Hook: '{cand['hook_preview'][:25]}'): "
                f"Feeding 30s slice to TRIBE v2..."
            )

        try:
            # 1. Fetch video slice (Instant from local cache or on-demand stream)
            if cached_video and os.path.exists(cached_video):
                # Instant 0.05s cut with local ffmpeg
                cmd_cut = [
                    "ffmpeg", "-y",
                    "-ss", str(cand["start"]),
                    "-t", str(cand["duration"]),
                    "-i", cached_video,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-c:a", "aac",
                    cand_slice_video
                ]
                subprocess.run(cmd_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif yt_url:
                download_youtube_video_section(
                    yt_url,
                    start_sec=cand["start"],
                    end_sec=cand["end"],
                    output_path=cand_slice_video
                )

            # Extract slice audio for acoustic scoring
            cmd_audio = [
                "ffmpeg", "-y",
                "-ss", str(cand["start"]),
                "-t", str(cand["duration"]),
                "-i", raw_audio_path,
                cand_slice_audio
            ]
            subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # 2. Feed slice to Meta TRIBE v2 / Neural Quality Gate
            slice_words = cand.get("words", [])
            rel_words = [
                {"word": w["word"], "start": max(0.0, w["start"] - cand["start"]), "end": w["end"] - cand["start"]}
                for w in slice_words
            ]

            score_data = scorer.score_slice(cand_slice_video, cand_slice_audio, rel_words)
            overall_score = score_data["overall_score"]

            cand_copy = dict(cand)
            cand_copy["neural_score"] = overall_score
            cand_copy["hook_score"] = score_data["hook_score"]
            cand_copy["mid_score"] = score_data["mid_score"]
            cand_copy["payoff_score"] = score_data["payoff_score"]
            cand_copy["curve"] = score_data["curve"]
            cand_copy["slice_video_path"] = cand_slice_video

            if os.path.exists(cand_slice_audio):
                try:
                    os.remove(cand_slice_audio)
                except OSError:
                    pass

            # 3. Decision Loop
            # Auto-approve when score clears threshold, after 5 attempts, or when no other candidates remain
            remaining_after_this = total_candidates - (i + 1)
            needed_now = target_count - len(accepted)
            is_only_option = (total_candidates <= target_count) or (len(accepted) == 0 and remaining_after_this == 0 and not rejected_pool)

            if overall_score >= current_threshold or tested_count >= 5 or is_only_option:
                # Clip approved!
                if progress_callback:
                    tag = "Best Available Match" if overall_score < current_threshold else "Neural Retention Score"
                    progress_callback(
                        f"APPROVED Clip #{len(accepted)+1}! {tag}: {overall_score}% "
                        f"(Hook Score: {score_data['hook_score']}%)"
                    )

                accepted.append(cand_copy)
                if len(accepted) >= target_count:
                    break  # Target achieved!
            else:
                if progress_callback:
                    progress_callback(
                        f"Candidate below threshold ({overall_score}% < {current_threshold}%). "
                        f"Checking next candidate (saving as backup if needed)..."
                    )
                rejected_pool.append(cand_copy)

        except Exception as e:
            print(f"Error testing candidate {cand['start']}: {e}")
            if os.path.exists(cand_slice_video):
                try:
                    os.remove(cand_slice_video)
                except Exception:
                    pass
            if os.path.exists(cand_slice_audio):
                try:
                    os.remove(cand_slice_audio)
                except Exception:
                    pass
            continue

    # 4. Graceful Fallback: if fewer than target_count cleared the strict gate,
    # promote the highest-scoring non-overlapping candidates from rejected_pool so we never fail with 0 clips
    if len(accepted) < target_count and rejected_pool:
        rejected_pool.sort(key=lambda c: c.get("neural_score", 0.0), reverse=True)
        for cand_copy in rejected_pool:
            if len(accepted) >= target_count:
                break
            if not is_overlapping(cand_copy["start"], cand_copy["end"], accepted):
                if progress_callback:
                    progress_callback(
                        f"APPROVED Clip #{len(accepted)+1} (Top Ranked Fallback: {cand_copy['neural_score']}%, "
                        f"Hook Score: {cand_copy['hook_score']}%)"
                    )
                accepted.append(cand_copy)

    # Clean up unused backup slice files
    accepted_paths = {c.get("slice_video_path") for c in accepted}
    for cand_copy in rejected_pool:
        p = cand_copy.get("slice_video_path")
        if p and p not in accepted_paths and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return accepted
