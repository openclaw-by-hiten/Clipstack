import os
import json
import uuid
import time
from core.config import EXPORTS_DIR, WORKSPACE_DIR, MAX_CLIPS_PER_SESSION
from core.ingestion import download_youtube_audio_only, process_local_file
from core.transcription import SpeechTranscriber, fetch_youtube_instant_subtitles
from core.neural_scorer import NeuralMomentScorer
from core.miner import mine_candidate_windows
from core.verifier import verify_and_select_clips_streaming
from core.smart_crop import generate_reframe_plan
from core.subtitle_engine import generate_ass_subtitles
from core.renderer import render_clip


class ClipPipelineOrchestrator:
    def __init__(self):
        self.transcriber = SpeechTranscriber(model_size="base")
        self.neural_scorer = NeuralMomentScorer()

    def process_session(self, request_params: dict, progress_callback=None) -> dict:
        """
        Executes Stream-Sliced, Audio-First pipeline:
        Step 1: Download lightweight audio via Cloudflare WARP Shield.
        Step 2: Instant timed subtitles (0.5s) or Faster-Whisper.
        Step 3: Mine sentence-snapped + acoustically-peaked viral moments across full video.
        Step 4: Stream unthrottled HLS HD slice & score with Neural Quality Gate.
        Step 5: AI Auto-Align (Face-Track / Duo Split / Cinema Smart-Fit) for 9:16, 1:1, or 16:9.
        Step 6: Burn safe-zone captions & export faststart web-optimized MP4 + clip-aligned metadata.
        """
        session_id = str(uuid.uuid4())[:8]
        yt_url = request_params.get("youtube_url")
        local_file = request_params.get("local_file")
        target_duration = int(request_params.get("target_duration", 60))
        clip_count = min(int(request_params.get("clip_count", 3)), MAX_CLIPS_PER_SESSION)
        aspect_ratio = request_params.get("aspect_ratio", "9:16")
        framing_mode = request_params.get("framing_mode", "auto")
        caption_theme = request_params.get("caption_theme", "hormozi")

        from core.branding import get_brand_config
        brand_cfg = get_brand_config()
        default_wm = brand_cfg.get("watermark_text") or brand_cfg.get("channel_handle") or "@clipstack_A1"
        watermark = request_params.get("watermark") or default_wm
        floating_watermark = request_params.get("floating_watermark", True)
        header_title = request_params.get("header_title")
        enable_vignette = request_params.get("enable_vignette", True)
        enable_grain = request_params.get("enable_grain", False)
        bgm_path = request_params.get("bgm_path")
        ai_thumbnail = request_params.get("ai_thumbnail", True)
        custom_thumbnail_path = request_params.get("custom_thumbnail_path")
        # ClipStack v3 Pro Engine flags (Phases 1-3)
        kill_dead_air = bool(request_params.get("kill_dead_air", False))
        word_align = bool(request_params.get("word_align", True))
        studio_audio = bool(request_params.get("studio_audio", True))
        strip_music = bool(request_params.get("strip_music", False))
        enable_visual_hype = bool(request_params.get("enable_visual_hype", True))
        enable_60fps = bool(request_params.get("enable_60fps", False))
        hook_voiceover_text = (request_params.get("hook_voiceover_text") or "").strip() or None

        def emit(msg: str, percent: int = 0):
            print(f"[{session_id}] ({percent}%) {msg}")
            if progress_callback:
                progress_callback({"message": msg, "percent": percent, "session_id": session_id})

        emit("STEP 1: Fetching lightweight audio stream (Zero full-video download)...", 8)

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

        words = []
        detected_lang = "en"

        if yt_url:
            emit("Checking for YouTube instant word-level subtitles...", 28)
            instant_data = fetch_youtube_instant_subtitles(yt_url)
            if instant_data and instant_data.get("words"):
                words = instant_data["words"]
                detected_lang = instant_data.get("language", "en")
                emit(f"Extracted {len(words)} timed words instantly from YouTube!", 48)

        if not words:
            emit("Transcribing audio with Faster-Whisper (greedy decoding)...", 30)

            def stt_callback(msg, pct):
                overall_pct = int(30 + (pct * 0.20))
                emit(msg, overall_pct)

            stt_result = self.transcriber.transcribe(audio_path, progress_callback=stt_callback)
            words = stt_result["words"]
            detected_lang = stt_result["language"]
            emit(f"Transcribed {len(words)} words [{detected_lang.upper()}].", 50)

        emit("Mining acoustic peaks & complete story-arc hooks across full video...", 55)
        candidates = mine_candidate_windows(
            words=words,
            video_duration=video_duration,
            target_duration=target_duration,
            clip_count=clip_count,
            audio_path=audio_path,
            source_title=video_title,
        )

        emit(f"Mined {len(candidates)} sentence-aligned candidate hooks. Starting Neural Verification...", 62)

        verified_clips = verify_and_select_clips_streaming(
            scorer=self.neural_scorer,
            yt_url=yt_url,
            local_video_path=local_video_path,
            raw_audio_path=audio_path,
            candidates=candidates,
            target_count=clip_count,
            target_duration=target_duration,
            progress_callback=lambda m: emit(m, 75),
            source_title=video_title,
            enable_visual_hype=enable_visual_hype,
        )

        if not verified_clips:
            raise RuntimeError("Could not find moments meeting the neural virality threshold.")

        emit(f"Quality Gate Complete: {len(verified_clips)} viral moments verified! Auto-aligning & rendering...", 82)

        rendered_outputs = []
        total_clips = len(verified_clips)

        for i, clip in enumerate(verified_clips, 1):
            clip_id = f"clip_{session_id}_{i}"
            c_start = clip["start"]
            c_end = clip["end"]
            c_dur = clip["duration"]
            orig_slice_source = clip["slice_video_path"]
            slice_source = orig_slice_source
            trimmed_dead_air_sec = 0.0

            # Phase 1: Optional Dead-Air / Silence Trimmer (auto-editor / FFmpeg silence killer)
            if kill_dead_air:
                emit(f"Trimming dead air & silent pauses on Clip {i}/{total_clips}...", 83)
                try:
                    from core.silence_trimmer import trim_dead_air
                    new_src, new_dur, saved_s = trim_dead_air(slice_source)
                    if saved_s > 0.0 and new_src != slice_source:
                        slice_source = new_src
                        c_dur = new_dur
                        trimmed_dead_air_sec = round(saved_s, 2)
                except Exception as e:
                    print(f"[Orchestrator] Dead-air trim fallback: {e}")

            # Phase 2: AI Auto-Align (PySceneDetect cuts + MediaPipe Lip Lock + YOLO11n + YuNet)
            reframe_plan = generate_reframe_plan(
                video_path=slice_source,
                clip_duration=c_dur,
                aspect_ratio=aspect_ratio,
                framing_mode=framing_mode,
            )
            layout_used = reframe_plan["layout_used"]

            step_percent = int(84 + (i / total_clips) * 14)
            emit(
                f"Syncing captions & framing Clip {i}/{total_clips} [{aspect_ratio} • {layout_used}]...",
                min(88, step_percent),
            )

            # Phase 1: Millisecond Word-Level Forced Alignment (WhisperX / Faster-Whisper on clip slice)
            ass_path = str(WORKSPACE_DIR / f"{clip_id}.ass")
            if word_align or trimmed_dead_air_sec > 0.0:
                try:
                    from core.alignment import align_clip_words
                    aligned_words = align_clip_words(
                        slice_video_path=slice_source,
                        fallback_words=words,
                        clip_start=c_start,
                        clip_end=c_end,
                        language=detected_lang,
                        force_retranscribe=(trimmed_dead_air_sec > 0.0),
                    )
                    generate_ass_subtitles(
                        words=aligned_words,
                        clip_start=0.0,
                        clip_end=c_dur + 0.25,
                        output_path=ass_path,
                        theme_key=caption_theme,
                        aspect_ratio=aspect_ratio,
                        margin_v_override=reframe_plan.get("caption_margin_v"),
                        alignment_override=reframe_plan.get("caption_alignment"),
                    )
                except Exception as e:
                    print(f"[Orchestrator] Word alignment fallback: {e}")
                    generate_ass_subtitles(
                        words=words,
                        clip_start=c_start,
                        clip_end=c_end,
                        output_path=ass_path,
                        theme_key=caption_theme,
                        aspect_ratio=aspect_ratio,
                        margin_v_override=reframe_plan.get("caption_margin_v"),
                        alignment_override=reframe_plan.get("caption_alignment"),
                    )
            else:
                generate_ass_subtitles(
                    words=words,
                    clip_start=c_start,
                    clip_end=c_end,
                    output_path=ass_path,
                    theme_key=caption_theme,
                    aspect_ratio=aspect_ratio,
                    margin_v_override=reframe_plan.get("caption_margin_v"),
                    alignment_override=reframe_plan.get("caption_alignment"),
                )

            smart_clip_title = (
                header_title.strip()
                if (header_title and header_title.strip())
                else " ".join(clip.get("hook_preview", f"Viral Highlight #{i}").split())
            )
            effective_header = header_title.strip() if (header_title and header_title.strip()) else None

            # Phase 3: Optional AI Spoken Hook Voiceover (Chatterbox / edge-tts)
            hook_vo_wav = None
            if hook_voiceover_text:
                try:
                    from core.hook_voiceover import generate_hook_voiceover
                    vo_path = str(WORKSPACE_DIR / f"{clip_id}_hook_vo.wav")
                    hook_vo_wav = generate_hook_voiceover(hook_voiceover_text, vo_path)
                except Exception as e:
                    print(f"[Orchestrator] Hook voiceover skip: {e}")

            clip_thumb_path = str(EXPORTS_DIR / f"{clip_id}_thumb.jpg")
            if custom_thumbnail_path and os.path.exists(custom_thumbnail_path):
                clip_custom_thumb = custom_thumbnail_path
            else:
                emit(
                    f"Generating 8K topic-matched AI thumbnail for Clip {i}/{total_clips}...",
                    min(93, step_percent),
                )
                from core.thumbnail_generator import generate_clip_thumbnail
                clip_custom_thumb = generate_clip_thumbnail(
                    video_path=slice_source,
                    clip_start=0.0,
                    clip_duration=c_dur,
                    output_thumb_path=clip_thumb_path,
                    clip_index=i,
                    hook_text=smart_clip_title,
                    transcript_text=clip.get("transcript", ""),
                    source_title=video_title,
                    aspect_ratio=aspect_ratio,
                    use_ai=ai_thumbnail,
                    visual_summary=clip.get("visual_summary"),
                )

            emit(
                f"Encoding Clip {i}/{total_clips} [{aspect_ratio} • {layout_used}{' • 60fps' if enable_60fps else ''}] + Studio Audio...",
                min(97, step_percent),
            )
            final_mp4 = str(EXPORTS_DIR / f"{clip_id}.mp4")
            render_clip(
                source_video=slice_source,
                clip_start=0.0,
                clip_duration=c_dur,
                ass_subtitle_path=ass_path,
                output_mp4=final_mp4,
                aspect_ratio=aspect_ratio,
                framing_mode=framing_mode,
                reframe_plan=reframe_plan,
                watermark=watermark,
                floating_watermark=floating_watermark,
                header_title=effective_header,
                enable_vignette=enable_vignette,
                enable_grain=enable_grain,
                bgm_path=bgm_path,
                custom_thumbnail_path=clip_custom_thumb,
                enable_60fps=enable_60fps,
                studio_audio=studio_audio,
                strip_music=strip_music,
                hook_voiceover_wav=hook_vo_wav,
            )

            for temp_p in {slice_source, orig_slice_source, hook_vo_wav}:
                if temp_p and os.path.exists(temp_p):
                    try:
                        os.remove(temp_p)
                    except Exception:
                        pass

            file_size_mb = round(os.path.getsize(final_mp4) / (1024 * 1024), 2) if os.path.exists(final_mp4) else 0.0

            meta = {
                "id": clip_id,
                "title": smart_clip_title,
                "source_title": video_title,
                "hook_line": clip.get("hook_line", smart_clip_title),
                "payoff_line": clip.get("payoff_line", ""),
                "transcript": clip.get("transcript", ""),
                "duration": c_dur,
                "start": c_start,
                "end": c_end,
                "aspect_ratio": aspect_ratio,
                "framing_mode": framing_mode,
                "layout": layout_used,
                "layout_used": layout_used,
                "file_size_mb": file_size_mb,
                "neural_score": clip.get("neural_score", 82.0),
                "hook_score": clip.get("hook_score", 84.0),
                "visual_score": clip.get("visual_score"),
                "visual_summary": clip.get("visual_summary", ""),
                "trimmed_dead_air_sec": trimmed_dead_air_sec,
                "enable_60fps": enable_60fps,
                "video_url": f"/api/stream/{clip_id}.mp4",
                "download_url": f"/api/download/{clip_id}.mp4",
                "thumb_url": f"/api/stream/{clip_id}_thumb.jpg",
                "created_at": time.time(),
            }

            with open(str(EXPORTS_DIR / f"{clip_id}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            rendered_outputs.append(meta)

        if yt_url:
            if local_video_path and os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                except Exception:
                    pass
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            for stale_wav in WORKSPACE_DIR.glob("temp_slice_*.wav"):
                try:
                    stale_wav.unlink()
                except OSError:
                    pass

        emit(f"Complete! {len(rendered_outputs)} verified clips exported & ready for instant streaming.", 100)

        return {
            "session_id": session_id,
            "source_title": video_title,
            "clips_count": len(rendered_outputs),
            "clips": rendered_outputs,
        }
