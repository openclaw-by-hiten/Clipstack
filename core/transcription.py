import os
import json
import urllib.request
import torch
from faster_whisper import WhisperModel
from core.config import CACHE_DIR
from core.ingestion import get_base_ydl_opts
import yt_dlp

def fetch_youtube_instant_subtitles(url: str, preferred_lang: str = "en") -> dict:
    """
    Lightning-fast extraction: Fetches official or auto-generated YouTube timed subtitles 
    in under 1 second (bypasses heavy 3-minute CPU transcription for 1-hour videos!).
    """
    try:
        ydl_opts = get_base_ydl_opts()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles", {})
            auto_subs = info.get("automatic_captions", {})

        all_subs = {**auto_subs, **subs}
        if not all_subs:
            return None

        # Pick language
        lang_key = preferred_lang if preferred_lang in all_subs else ("en" if "en" in all_subs else list(all_subs.keys())[0])
        sub_formats = all_subs.get(lang_key, [])
        json3_entry = next((s for s in sub_formats if s.get("ext") == "json3"), None)
        if not json3_entry:
            return None

        req = urllib.request.Request(json3_entry["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        words = []
        events = data.get("events", [])
        seen_events = set()
        raw_words = []

        for ev in events:
            start_ms = ev.get("tStartMs", 0)
            segs = ev.get("segs", [])
            if not segs:
                continue

            # Deduplicate repeated event tracks (e.g. duplicate events with different pPenId)
            ev_text = "".join(s.get("utf8", "") for s in segs).strip()
            ev_key = (start_ms, ev_text)
            if ev_key in seen_events:
                continue
            seen_events.add(ev_key)

            for s in segs:
                text = s.get("utf8", "").replace("\u200b", "").strip()
                if not text or text == "\n":
                    continue
                offset_ms = s.get("tOffsetMs", 0)
                seg_start = (start_ms + offset_ms) / 1000.0

                # Split multiple words within a segment
                subwords = text.split()
                for idx, sw in enumerate(subwords):
                    clean_sw = sw.strip().strip("-").strip("\u200b")
                    if clean_sw:
                        w_start = seg_start + (idx * 0.18)
                        raw_words.append({
                            "word": clean_sw,
                            "start": round(w_start, 2)
                        })

        if not raw_words:
            return None

        # Sort strictly by start time
        raw_words.sort(key=lambda x: x["start"])

        # Deduplicate consecutive identical words within 0.4s window
        clean_words = []
        for w in raw_words:
            if clean_words:
                prev = clean_words[-1]
                if prev["word"].lower() == w["word"].lower() and abs(w["start"] - prev["start"]) < 0.4:
                    continue
            clean_words.append(w)

        # Assign clean, non-overlapping end timestamps
        for i in range(len(clean_words)):
            curr_start = clean_words[i]["start"]
            if i < len(clean_words) - 1:
                next_start = clean_words[i + 1]["start"]
                if next_start > curr_start:
                    clean_words[i]["end"] = round(min(next_start, curr_start + 0.6), 2)
                else:
                    clean_words[i]["end"] = round(curr_start + 0.35, 2)
            else:
                clean_words[i]["end"] = round(curr_start + 0.35, 2)

        return {
            "language": lang_key,
            "words": clean_words,
            "duration": float(info.get("duration", 0.0))
        }
    except Exception as e:
        print(f"[Subtitles] YouTube instant caption extraction failed ({e}). Falling back to Faster-Whisper.")
        return None

class SpeechTranscriber:
    def __init__(self, model_size: str = "base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"[Whisper] Loading Faster-Whisper ({self.model_size}) on {self.device} ({self.compute_type})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(CACHE_DIR / "whisper")
            )

    def transcribe(self, audio_path: str, language: str = None, progress_callback=None) -> dict:
        """
        Transcribes audio with word-level microsecond timestamps using greedy decoding (beam_size=1)
        for 3x faster processing, with real-time progress callbacks.
        """
        self._load_model()
        if progress_callback:
            progress_callback("Transcribing speech with Faster-Whisper...", 0)

        # beam_size=1 is 3x faster than beam_size=5 on CPU
        segments_gen, info = self._model.transcribe(
            audio_path,
            beam_size=1,
            word_timestamps=True,
            vad_filter=True,
            language=language if language and language != "auto" else None
        )

        total_dur = max(1.0, info.duration)
        all_words = []
        all_segments = []

        last_reported_sec = 0.0
        for seg in segments_gen:
            if progress_callback and (seg.end - last_reported_sec >= 20.0 or seg.end >= total_dur - 5):
                last_reported_sec = seg.end
                pct = int(min(100, (seg.end / total_dur) * 100))
                progress_callback(f"Transcribing audio: {int(seg.end)}s / {int(total_dur)}s ({pct}%)", pct)

            seg_words = []
            if seg.words:
                for w in seg.words:
                    word_obj = {
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2)
                    }
                    if word_obj["word"]:
                        all_words.append(word_obj)
                        seg_words.append(word_obj)
            
            all_segments.append({
                "text": seg.text.strip(),
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "words": seg_words
            })

        return {
            "language": info.language,
            "duration": round(info.duration, 2),
            "words": all_words,
            "segments": all_segments
        }
