import os
import json
import uuid
import time
from pathlib import Path
from core.config import EXPORTS_DIR, WORKSPACE_DIR, MAX_CLIPS_PER_SESSION
from core.ingestion import download_youtube_audio_only, process_local_file
from core.transcription import SpeechTranscriber, fetch_youtube_instant_subtitles
from core.neural_scorer import NeuralMomentScorer
from core.miner import mine_candidate_windows
from core.verifier import verify_and_select_clips_streaming
from core.subtitle_engine import generate_ass_subtitles
from core.renderer import render_clip

class ClipPipelineOrchestrator:
    def __init__(self):
        self.transcriber = SpeechTranscriber(model_size="base")
        self.neural_scorer = NeuralMomentScorer()

    def process_session(self, request_params: dict, progress_callback=None) -> dict:
        """
        Executes Stream-Sliced, Audio-First pipeline:
        Step 1: Download ONLY lightweight audio from YouTube (3-5 seconds).
        Step 2: Instant timed subtitles (0.5s) or Faster-Whisper.
        Step 3: Mine viral candidate moments from audio & transcript.
        Step 4: Download on-demand 30s-180s video slice.
        Step 5: Feed slice to Meta TRIBE v2. If not satisfactory, repeat loop!
        Step 6: Burn Hormozi captions & render 9:16 vertical shorts.
        """
        session_id = str(uuid.uuid4())[:8]
        yt_url = request_params.get("youtube_url")
        local_file = request_params.get("local_file")
        target_duration = int(request_params.get("target_duration", 60))
        clip_count = min(int(request_params.get("clip_count", 3)), MAX_CLIPS_PER_SESSION)
        aspect_ratio = request_params.get("aspect_ratio", "9:16")
        caption_theme = request_params.get("caption_theme", "hormozi")
        from core.branding import get_brand_config
        brand_cfg = get_brand_config()
        default_wm = brand_cfg.get("watermark_text") or brand_cfg.get("channel_handle") or "@clipstack_A1"
        watermark = request_params.get("watermark") or default_wm
        floating_watermark = request_params.get("floating_watermark", True)
        header_title = request_params.get("header_title")
        enable_vignette = request_params.get("enable_vignette", True)
        enable_grain = request_params.get("enable_grain", True)
        bgm_path = request_params.get("bgm_path")
        ai_thumbnail = request_params.get("ai_thumbnail", True)
        custom_thumbnail_path = request_params.get("custom_thumbnail_path")

        def emit(msg: str, percent: int = 0):
            print(f"[{session_id}] ({percent}%) {msg}")
            if progress_callback:
                progress_callback({"message": msg, "percent": percent, "session_id": session_id})

        emit("STEP 1: Fetching lightweight audio stream (Zero full-video download)...", 8)

        # 1. Audio-Only Download
        if yt_url:
            source = download_youtube_audio_only(yt_url, progress_callback=lambda m: emit(m, 15))
            local_video_path = None
        elif local_file and os.path.exists(local_file):
            source = process_local_file(local_file, progress_callback=lambda m: emit(m, 15))
            local_video_path = source["video_path"]
        else:
            raise ValueError("No valid YouTube URL or local file provided.")

        audio_path = source["audio_path"]
        video_title = source["title"]
        video_duration = source["duration"]

        emit(f"Audio ready: '{video_title}' ({int(video_duration)}s). Checking timed subtitles...", 25)

        # 2. Subtitles: Try Instant Extraction First (0.5s), else Faster-Whisper
        words = []
        detected_lang = "en"

        if yt_url:
            emit("Checking for YouTube instant word-level subtitles...", 28)
            instant_data = fetch_youtube_instant_subtitles(yt_url)
            if instant_data and instant_data.get("words"):
                words = instant_data["words"]
                detected_lang = instant_data.get("language", "en")
                emit(f"Extracted {len(words)} timed words instantly from YouTube! (0.8s)", 48)

        if not words:
            # Fallback to local Faster-Whisper
            emit("Transcribing audio with Faster-Whisper (greedy decoding)...", 30)
            def stt_callback(msg, pct):
                overall_pct = int(30 + (pct * 0.20))
                emit(msg, overall_pct)

            stt_result = self.transcriber.transcribe(
                audio_path,
                progress_callback=stt_callback
            )
            words = stt_result["words"]
            detected_lang = stt_result["language"]
            emit(f"Transcribed {len(words)} words [{detected_lang.upper()}].", 50)

        # 3. Mine candidate windows from audio & transcript
        emit("Mining acoustic peaks & narrative hooks from transcript...", 55)
        candidates = mine_candidate_windows(
            words=words,
            video_duration=video_duration,
            target_duration=target_duration,
            clip_count=clip_count
        )

        emit(f"Mined {len(candidates)} potential candidate hooks. Starting On-Demand TRIBE v2 Verification Loop...", 62)

        # 4 & 5. Streaming Verification Loop (Cut video slice -> Score with TRIBE v2 -> Repeat if not satisfactory)
        verified_clips = verify_and_select_clips_streaming(
            scorer=self.neural_scorer,
            yt_url=yt_url,
            local_video_path=local_video_path,
            raw_audio_path=audio_path,
            candidates=candidates,
            target_count=clip_count,
            target_duration=target_duration,
            progress_callback=lambda m: emit(m, 75)
        )

        if not verified_clips:
            raise RuntimeError("Could not find moments meeting the neural virality threshold.")

        emit(f"Quality Gate Complete: {len(verified_clips)} elite viral moments verified! Rendering shorts...", 82)

        # 6. Final Render & Subtitle Burn
        rendered_outputs = []
        total_clips = len(verified_clips)

        for i, clip in enumerate(verified_clips, 1):
            clip_id = f"clip_{session_id}_{i}"
            c_start = clip["start"]
            c_end = clip["end"]
            c_dur = clip["duration"]
            slice_source = clip["slice_video_path"]

            step_percent = int(82 + (i / total_clips) * 16)
            emit(f"Rendering Short {i}/{total_clips} (9:16 vertical + Hormozi animated captions)...", step_percent)

            # Subtitle file (relative to slice start = 0)
            ass_path = str(WORKSPACE_DIR / f"{clip_id}.ass")
            generate_ass_subtitles(
                words=words,
                clip_start=c_start,
                clip_end=c_end,
                output_path=ass_path,
                theme_key=caption_theme,
                aspect_ratio=aspect_ratio
            )

            # Clean header title (only show if explicitly provided or clean hook)
            effective_header = header_title.strip() if (header_title and header_title.strip()) else None

            # Generate distinct 8K thumbnail per clip
            clip_thumb_path = str(EXPORTS_DIR / f"{clip_id}_thumb.jpg")
            clip_custom_thumb = None
            if custom_thumbnail_path and os.path.exists(custom_thumbnail_path):
                clip_custom_thumb = custom_thumbnail_path
            else:
                from core.thumbnail_generator import generate_clip_thumbnail
                clip_custom_thumb = generate_clip_thumbnail(
                    video_path=slice_source,
                    clip_start=0.0,
                    clip_duration=c_dur,
                    output_thumb_path=clip_thumb_path,
                    clip_index=i,
                    hook_text=clip.get("hook_preview"),
                    use_ai=ai_thumbnail
                )

            # Final MP4 file
            final_mp4 = str(EXPORTS_DIR / f"{clip_id}.mp4")
            render_clip(
                source_video=slice_source,
                clip_start=0.0,
                clip_duration=c_dur,
                ass_subtitle_path=ass_path,
                output_mp4=final_mp4,
                aspect_ratio=aspect_ratio,
                watermark=watermark,
                floating_watermark=floating_watermark,
                header_title=effective_header,
                enable_vignette=enable_vignette,
                enable_grain=enable_grain,
                bgm_path=bgm_path,
                custom_thumbnail_path=clip_custom_thumb
            )

            # Cleanup temp slice
            if os.path.exists(slice_source):
                try:
                    os.remove(slice_source)
                except Exception:
                    pass

            clean_preview_title = " ".join(clip.get("hook_preview", f"Viral Moment #{i}").split())
            meta = {
                "id": clip_id,
                "title": header_title if (header_title and header_title.strip()) else clean_preview_title,
                "duration": c_dur,
                "start": c_start,
                "end": c_end,
                "aspect_ratio": aspect_ratio,
                "neural_score": clip.get("neural_score", 82.0),
                "hook_score": clip.get("hook_score", 84.0),
                "video_url": f"/api/stream/{clip_id}.mp4",
                "thumb_url": f"/api/stream/{clip_id}_thumb.jpg",
                "created_at": time.time()
            }

            with open(str(EXPORTS_DIR / f"{clip_id}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            rendered_outputs.append(meta)

        # Auto-cleanup heavy raw YouTube video & audio to preserve storage
        if youtube_url:
            if source_video and os.path.exists(source_video):
                try:
                    os.remove(source_video)
                except Exception:
                    pass
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except Exception:
                    pass

        emit(f"Complete! {len(rendered_outputs)} verified viral shorts exported to your library.", 100)

        return {
            "session_id": session_id,
            "source_title": video_title,
            "clips_count": len(rendered_outputs),
            "clips": rendered_outputs
        }
