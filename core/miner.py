import os
import re
import wave
import numpy as np
from typing import List, Dict, Optional

HOOK_STARTERS = {
    "if", "why", "how", "what", "when", "imagine", "watch", "look",
    "nobody", "everyone", "never", "always", "stop", "wait", "secret",
    "truth", "crazy", "insane", "heres", "here's", "listen", "most",
}

HIGH_IMPACT_WORDS = {
    "million": 9, "billion": 9, "thousand": 7, "dollar": 8, "dollars": 8,
    "money": 7, "cash": 7, "prize": 8, "win": 8, "won": 8, "winner": 8,
    "lose": 7, "lost": 7, "pounds": 7, "lbs": 7, "weight": 6,
    "challenge": 7, "impossible": 8, "insane": 8, "crazy": 7, "shocking": 8,
    "secret": 8, "mistake": 8, "truth": 7, "never": 6, "quit": 7,
    "eliminated": 8, "survived": 8, "final": 6, "finally": 7, "moment": 6,
    "changed": 6, "life": 5, "everything": 6, "hardest": 7, "biggest": 7,
    "worst": 7, "best": 6, "rule": 6, "rules": 6, "days": 5, "hours": 5,
}


def compute_fast_audio_rms_1hz(audio_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Computes 1-second normalized RMS loudness across the entire 16kHz mono WAV
    in ~0.12s using stdlib wave + numpy (zero librosa overhead on 1-hour files).
    """
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate() or 16000
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            if sampwidth != 2 or n_frames == 0:
                return None
            raw = wf.readframes(n_frames)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            samples = samples[::n_channels]

        n_secs = len(samples) // sr
        if n_secs < 2:
            return None
        trimmed = samples[: n_secs * sr].reshape(n_secs, sr)
        rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
        p5, p98 = np.percentile(rms, 5), np.percentile(rms, 98)
        norm = np.clip((rms - p5) / max(1.0, p98 - p5), 0.0, 1.0)
        return norm
    except Exception:
        return None


def find_sentence_starts(words: List[Dict]) -> List[int]:
    """Indices of words that begin a fresh sentence or thought after punctuation/pause."""
    starts = [0]
    for i in range(1, len(words)):
        prev_word = words[i - 1]["word"]
        pause = words[i]["start"] - words[i - 1]["end"]
        if prev_word.endswith((".", "!", "?")) or pause >= 0.45:
            starts.append(i)
    return starts


def find_sentence_ends(words: List[Dict]) -> List[int]:
    """Indices of words that complete a sentence or thought cleanly."""
    ends = []
    for i in range(len(words) - 1):
        curr_word = words[i]["word"]
        pause = words[i + 1]["start"] - words[i]["end"]
        if curr_word.endswith((".", "!", "?")) or pause >= 0.42:
            ends.append(i)
    if words:
        ends.append(len(words) - 1)
    return ends


def _clean_sentence_text(word_objs: List[Dict], max_words: int = 16) -> str:
    tokens = [w["word"].strip() for w in word_objs[:max_words] if w.get("word", "").strip()]
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()


def build_clip_smart_title(window_words: List[Dict], source_title: str = "") -> dict:
    """
    Synthesizes a complete, human-readable, social-ready title + hook_line + payoff_line
    directly from the clip's actual spoken sentences (never chopping mid-word like 'lose 1...').
    """
    if not window_words:
        fallback = source_title or "Viral Highlight"
        return {
            "title": fallback[:75],
            "hook_line": fallback,
            "payoff_line": "",
            "transcript": "",
        }

    transcript = _clean_sentence_text(window_words, max_words=len(window_words))

    # Split transcript into clean sentences
    sentences = []
    curr = []
    for i, w in enumerate(window_words):
        curr.append(w)
        is_last = i == len(window_words) - 1
        pause = (window_words[i + 1]["start"] - w["end"]) if not is_last else 1.0
        if w["word"].endswith((".", "!", "?")) or pause >= 0.55 or len(curr) >= 18 or is_last:
            s_txt = _clean_sentence_text(curr, max_words=22)
            alpha_words = [t for t in re.findall(r"[A-Za-z]+", s_txt) if len(t) >= 2]
            if len(s_txt.split()) >= 4 and len(alpha_words) >= 3:
                sentences.append(s_txt)
            curr = []

    hook_line = sentences[0] if sentences else _clean_sentence_text(window_words[:14], 14)
    payoff_line = sentences[-1] if len(sentences) > 1 else ""

    # Pick the punchiest complete sentence in the clip for the headline (numbers, stakes, ?, !, high-impact words)
    best_sent = hook_line
    best_sent_score = -1.0
    for idx, s in enumerate(sentences[:6]):
        sl = s.lower()
        wc = len(s.split())
        if wc < 4:
            continue
        sc = 4.0 if idx == 0 else (2.0 if idx == 1 else 0.0)
        if re.search(r"(\$?\d[\d,]*)", s):
            sc += 5.5
        if "!" in s or "?" in s:
            sc += 2.5
        for kw, wgt in HIGH_IMPACT_WORDS.items():
            if re.search(rf"\b{re.escape(kw)}\b", sl):
                sc += wgt * 0.5
        if 5 <= wc <= 16:
            sc += 3.5
        if sc > best_sent_score:
            best_sent_score = sc
            best_sent = s

    # Clean trailing filler/dangling words and preserve complete punchlines up to 92 chars
    headline = best_sent.strip(" ,;:-")
    dangling_tail = {
        "and", "or", "but", "so", "because", "that", "which", "the", "a", "an",
        "to", "of", "in", "with", "for", "from", "at", "by", "on", "before", "after",
        "when", "while", "if", "they", "we", "you", "he", "she", "it", "i",
        "their", "our", "your", "his", "her", "my", "can", "will", "would", "could", "should",
    }
    words_h = headline.split()
    while len(words_h) > 4 and words_h[-1].lower().strip(".,!?") in dangling_tail:
        words_h.pop()

    joined_h = " ".join(words_h)
    if len(joined_h) > 92:
        # If there's a clause comma with a concise payoff clause (e.g. ", they win $250,000."), keep the payoff!
        clauses_h = re.split(r",\s+", joined_h)
        if len(clauses_h) >= 2:
            left_c, right_c = ", ".join(clauses_h[:-1]), clauses_h[-1].strip()
            if 6 <= len(right_c) <= 34:
                left_words = left_c.split()
                # Drop leading conditional setup ("If this couple can ") if needed
                if len(left_words) > 6 and left_words[0].lower() in ("if", "when", "while"):
                    for k, lw in enumerate(left_words[:5]):
                        if lw.lower() in ("can", "could", "will", "would"):
                            left_words = left_words[k + 1:]
                            break
                candidate_h = f"{' '.join(left_words).strip()}, {right_c}"
                if len(candidate_h) <= 92:
                    joined_h = candidate_h
        if len(joined_h) > 92:
            trimmed = []
            for wd in joined_h.split():
                if len(" ".join(trimmed + [wd])) > 88:
                    break
                trimmed.append(wd)
            while len(trimmed) > 4 and trimmed[-1].lower().strip(".,!?") in dangling_tail:
                trimmed.pop()
            headline = " ".join(trimmed).rstrip(",;:-")
        else:
            headline = joined_h
    else:
        headline = joined_h

    if headline and headline[0].islower():
        headline = headline[0].upper() + headline[1:]

    return {
        "title": headline,
        "hook_line": hook_line,
        "payoff_line": payoff_line,
        "transcript": transcript,
    }


def mine_candidate_windows(
    words: List[Dict],
    video_duration: float,
    target_duration: int,
    clip_count: int,
    step_sec: int = None,
    audio_path: Optional[str] = None,
    source_title: str = "",
) -> List[Dict]:
    """
    Scans the entire long-form video using both linguistic hook/story signals AND
    1Hz acoustic energy peaks, snapping BOTH clip start and clip end to natural
    sentence boundaries, and diversifying across the full storyline.
    """
    if not words or video_duration <= target_duration:
        meta = build_clip_smart_title(words, source_title=source_title)
        return [{
            "start": 0.0,
            "end": min(float(target_duration), video_duration),
            "duration": min(float(target_duration), video_duration),
            "hook_preview": meta["title"],
            "hook_line": meta["hook_line"],
            "payoff_line": meta["payoff_line"],
            "transcript": meta["transcript"],
            "words": words,
            "pre_score": 95.0,
        }]

    if step_sec is None:
        step_map = {30: 6, 60: 10, 90: 14, 180: 20}
        step_sec = step_map.get(target_duration, 10)

    rms_1hz = compute_fast_audio_rms_1hz(audio_path)
    sentence_starts = find_sentence_starts(words)
    sentence_ends = find_sentence_ends(words)

    start_times = np.array([words[idx]["start"] for idx in sentence_starts])
    end_times = np.array([words[idx]["end"] for idx in sentence_ends])

    raw_candidates = []
    seen_starts = set()
    max_start = max(0, int(video_duration - target_duration * 0.85))

    for rough_start in range(0, max_start + 1, step_sec):
        # 1. Snap start to nearest clean sentence start within +/- 4.0s
        diffs = np.abs(start_times - rough_start)
        best_s_pos = int(np.argmin(diffs))
        closest_snap = float(start_times[best_s_pos]) if diffs[best_s_pos] <= 4.0 else float(rough_start)

        snap_bucket = round(closest_snap, 1)
        if snap_bucket in seen_starts:
            continue
        seen_starts.add(snap_bucket)

        # 2. Snap end to nearest clean sentence completion within [target - 5s, target + 5s]
        ideal_end = min(video_duration, closest_snap + target_duration)
        end_diffs = np.abs(end_times - ideal_end)
        best_e_pos = int(np.argmin(end_diffs))
        if end_diffs[best_e_pos] <= 5.0 and (end_times[best_e_pos] - closest_snap) >= max(12.0, target_duration * 0.80):
            snapped_end = min(video_duration, float(end_times[best_e_pos]) + 0.15)
        else:
            snapped_end = ideal_end

        actual_dur = round(snapped_end - closest_snap, 2)
        if actual_dur < max(10.0, target_duration * 0.75):
            continue

        window_words = [w for w in words if closest_snap <= w["start"] <= snapped_end]
        if len(window_words) < (actual_dur * 0.75):
            continue

        # --- Multimodal Viral Moment Scoring ---
        # A. Opening 6-second Hook Power (crucial for Shorts 3-second swipe retention!)
        opening_words = [w for w in window_words if w["start"] <= closest_snap + 6.0]
        opening_text = " ".join(w["word"].lower().strip(".,!?\"'") for w in opening_words)
        first_token = opening_words[0]["word"].lower().strip(".,!?\"'") if opening_words else ""

        opening_score = 0.0
        if first_token in HOOK_STARTERS:
            opening_score += 10.0
        if any( ("?" in w["word"] or "!" in w["word"]) for w in opening_words ):
            opening_score += 7.0
        if re.search(r"\b\d+\b", opening_text):
            opening_score += 8.0

        # B. High-impact stakes & curiosity keywords across window
        matched_kws = set()
        kw_score = 0.0
        number_hits = 0
        for w in window_words:
            wl = w["word"].lower().strip(".,!?\"'$")
            if wl.isdigit():
                number_hits += 1
            for k, weight in HIGH_IMPACT_WORDS.items():
                if wl == k and k not in matched_kws:
                    matched_kws.add(k)
                    kw_score += weight

        kw_score += min(12.0, number_hits * 3.0)

        # C. Payoff / Climax in final 8 seconds
        closing_words = [w for w in window_words if w["start"] >= snapped_end - 8.0]
        payoff_score = 5.0 if (closing_words and closing_words[-1]["word"].endswith((".", "!", "?"))) else 0.0
        if any("!" in w["word"] for w in closing_words):
            payoff_score += 5.0

        # D. Acoustic Energy & Dynamic Excitement from real audio waveform
        acoustic_score = 0.0
        if rms_1hz is not None:
            s_i = max(0, min(len(rms_1hz) - 1, int(closest_snap)))
            e_i = max(s_i + 1, min(len(rms_1hz), int(snapped_end)))
            seg_rms = rms_1hz[s_i:e_i]
            if len(seg_rms) > 0:
                mean_e = float(np.mean(seg_rms))
                peak_e = float(np.percentile(seg_rms, 90))
                hook_e = float(np.mean(seg_rms[: min(5, len(seg_rms))]))
                acoustic_score = (mean_e * 18.0) + (peak_e * 14.0) + (hook_e * 10.0)

        # E. Pacing & Lexical Diversity
        total_tokens = len(window_words)
        unique_tokens = len(set(w["word"].lower() for w in window_words))
        lexical_diversity = unique_tokens / max(1, total_tokens)
        wpm = total_tokens / max(0.25, actual_dur / 60.0)
        pacing_score = min(22.0, wpm * 0.12)
        rep_penalty = 35.0 if lexical_diversity < 0.42 else 0.0

        # Mild premise bonus for strong intros without starving mid/late climaxes!
        premise_bonus = 6.0 if closest_snap <= 20 else 0.0
        # Avoid dead outro/sponsor tail in last 3% of long videos unless huge score
        outro_penalty = 18.0 if (video_duration > 300 and snapped_end > video_duration * 0.96) else 0.0

        pre_score = (
            opening_score
            + (kw_score * 1.35)
            + payoff_score
            + acoustic_score
            + pacing_score
            + premise_bonus
            - rep_penalty
            - outro_penalty
        )

        meta = build_clip_smart_title(window_words, source_title=source_title)

        raw_candidates.append({
            "start": round(closest_snap, 2),
            "end": round(snapped_end, 2),
            "duration": actual_dur,
            "hook_preview": meta["title"],
            "hook_line": meta["hook_line"],
            "payoff_line": meta["payoff_line"],
            "transcript": meta["transcript"],
            "words": window_words,
            "pre_score": round(pre_score, 2),
        })

    if not raw_candidates:
        meta = build_clip_smart_title(words, source_title=source_title)
        return [{
            "start": 0.0,
            "end": min(float(target_duration), video_duration),
            "duration": min(float(target_duration), video_duration),
            "hook_preview": meta["title"],
            "hook_line": meta["hook_line"],
            "payoff_line": meta["payoff_line"],
            "transcript": meta["transcript"],
            "words": words,
            "pre_score": 80.0,
        }]

    raw_candidates.sort(key=lambda x: x["pre_score"], reverse=True)

    # Diversify top candidates across narrative zones so multi-clip runs cover the whole video story!
    if video_duration > target_duration * 3:
        num_zones = max(3, min(6, clip_count + 2))
        zone_len = video_duration / num_zones
        zoned = [[] for _ in range(num_zones)]
        for c in raw_candidates:
            z_idx = min(num_zones - 1, int(c["start"] // zone_len))
            zoned[z_idx].append(c)

        # Round-robin pick the #1 best peak from each zone sorted by that zone's peak score, then remaining
        zone_order = sorted(
            range(num_zones),
            key=lambda z: zoned[z][0]["pre_score"] if zoned[z] else -999.0,
            reverse=True,
        )
        diversified = []
        max_depth = max((len(z) for z in zoned), default=0)
        for depth in range(max_depth):
            for z in zone_order:
                if depth < len(zoned[z]):
                    diversified.append(zoned[z][depth])
        return diversified

    return raw_candidates
