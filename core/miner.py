import math
from typing import List, Dict

HOOK_KEYWORDS_WEIGHTED = {
    "million": 10, "dollar": 8, "dollars": 8, "bought": 9, "entire": 7,
    "win": 8, "winner": 8, "won": 8, "crazy": 7, "insane": 7,
    "challenge": 7, "experiment": 7, "rules": 6, "everything": 6,
    "city": 5, "calories": 5, "impossible": 6, "prize": 7, "cash": 7,
    "secret": 6, "shocking": 6, "giveaway": 8, "eliminated": 7, "quit": 6
}

def find_sentence_starts(words: List[Dict]) -> List[int]:
    """Finds indices of words that represent natural sentence starts (after pauses or punctuation)."""
    starts = [0]
    for i in range(1, len(words)):
        prev_word = words[i - 1]["word"]
        pause_duration = words[i]["start"] - words[i - 1]["end"]
        if prev_word.endswith((".", "!", "?")) or pause_duration >= 0.5:
            starts.append(i)
    return starts

def mine_candidate_windows(
    words: List[Dict],
    video_duration: float,
    target_duration: int,
    clip_count: int,
    step_sec: int = None
) -> List[Dict]:
    """
    Scans the transcript to generate clean candidate windows matching target_duration.
    Ranks candidates in descending order of viral excitement so the most energetic,
    high-retention moments are tested FIRST.
    """
    if not words or video_duration <= target_duration:
        return [{
            "start": 0.0,
            "end": min(float(target_duration), video_duration),
            "duration": min(float(target_duration), video_duration),
            "hook_preview": "Full segment...",
            "words": words,
            "pre_score": 100.0
        }]

    if step_sec is None:
        step_map = {30: 8, 60: 12, 90: 16, 180: 24}
        step_sec = step_map.get(target_duration, 10)

    sentence_start_indices = find_sentence_starts(words)
    candidates = []
    max_start = max(0, int(video_duration - target_duration))

    for start_t in range(0, max_start + 1, step_sec):
        end_t = start_t + target_duration

        # Grab words in this window
        window_words = [w for w in words if start_t <= w["start"] <= end_t]
        if len(window_words) < (target_duration * 0.8):  # Skip silent/empty parts
            continue

        # Snap start to nearest sentence start within +/- 3.5s
        closest_snap = start_t
        for idx in sentence_start_indices:
            w_time = words[idx]["start"]
            if abs(w_time - start_t) <= 3.5:
                closest_snap = w_time
                break

        snapped_end = min(video_duration, closest_snap + target_duration)
        window_words = [w for w in words if closest_snap <= w["start"] <= snapped_end]
        if not window_words:
            continue

        hook_preview = " ".join([w["word"] for w in window_words[:8]])
        if hook_preview:
            hook_preview += "..."

        # Calculate Virality Pre-Score:
        # 1. Unique weighted keyword hits
        matched_kws = set()
        kw_score = 0
        for w in window_words:
            wl = w["word"].lower().strip(".!,?")
            for k, weight in HOOK_KEYWORDS_WEIGHTED.items():
                if k in wl and k not in matched_kws:
                    matched_kws.add(k)
                    kw_score += weight

        # 2. Lexical diversity check (penalizes spam chants like "eat eat eat")
        total_tokens = len(window_words)
        unique_tokens = len(set(w["word"].lower() for w in window_words))
        lexical_diversity = unique_tokens / max(1, total_tokens)

        # 3. Pacing & excitement
        wpm = total_tokens / (target_duration / 60.0)
        exclams = sum(1 for w in window_words if "!" in w["word"] or "?" in w["word"])
        pacing_score = min(30.0, wpm * 0.15) + min(15.0, exclams * 1.5)

        # 4. Opening hook bonus: The premise / hook of the video has the highest retention
        hook_bonus = 40.0 if closest_snap <= 16 else (15.0 if closest_snap <= 90 else 0.0)

        # 5. Repetition penalty if diversity is flat
        rep_penalty = 35.0 if lexical_diversity < 0.45 else 0.0

        pre_score = (kw_score * 2.0) + pacing_score + hook_bonus - rep_penalty

        candidates.append({
            "start": round(closest_snap, 2),
            "end": round(snapped_end, 2),
            "duration": round(snapped_end - closest_snap, 2),
            "hook_preview": hook_preview,
            "words": window_words,
            "pre_score": round(pre_score, 2)
        })

    # Sort descending: Highest viral hook score tested FIRST!
    candidates.sort(key=lambda x: x["pre_score"], reverse=True)
    return candidates
