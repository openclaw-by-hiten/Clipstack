import sys
import json
from core.orchestrator import ClipPipelineOrchestrator

def main():
    orch = ClipPipelineOrchestrator()
    params = {
        "youtube_url": "https://youtu.be/plN7JMbadRg",
        "target_duration": 30,
        "clip_count": 1,
        "aspect_ratio": "9:16",
        "caption_theme": "raj_shamani",
        "watermark": "@autoclip",
        "header_title": "I Bought An Entire Grocery Store!",
        "enable_vignette": True,
        "enable_grain": True,
        "custom_thumbnail_path": r"C:\Users\hiten\.gemini\antigravity\brain\22e8a20d-5839-42b1-84a2-c7cc1a0bed44\mrbeast_grocery_thumb_1791710242496.jpg"
    }
    print("[PIPELINE RUNNER] Starting autonomous job for 1x 30s 9:16 Raj Shamani viral clip...")
    def on_prog(d):
        print(f"[{d.get('percent', 0)}%] {d.get('message', '')}", flush=True)

    result = orch.process_session(params, progress_callback=on_prog)
    print("\n[JOB COMPLETED SUCCESSFULLY!]")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
