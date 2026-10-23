import sys
import os
import time

sys.path.append(r"C:\Users\hiten\.gemini\antigravity\scratch\neural-clip-studio")

from core.orchestrator import ClipPipelineOrchestrator
from core.config import EXPORTS_DIR

print(f"[{time.strftime('%X')}] Starting Local Pipeline Orchestration...")
print(f"Exports Directory: {EXPORTS_DIR}")

params = {
    "youtube_url": "https://youtu.be/RE3_PGwDwhw",
    "target_duration": 60,
    "clip_count": 3,
    "aspect_ratio": "9:16",
    "caption_theme": "hormozi",
    "watermark": "@clipstack_A1",
    "floating_watermark": True,
    "header_title": None,
    "enable_vignette": True,
    "enable_grain": True,
    "ai_thumbnail": True
}

def progress(data):
    pct = data.get("percent", 0)
    msg = data.get("message", "")
    print(f"[{time.strftime('%X')}] ({pct}%) {msg}", flush=True)

try:
    orch = ClipPipelineOrchestrator()
    result = orch.process_session(params, progress_callback=progress)
    print(f"\n[{time.strftime('%X')}] 🎉 SUCCESS! Result:", result)
    print("\nGenerated files in exports:")
    for f in os.listdir(EXPORTS_DIR):
        fp = os.path.join(EXPORTS_DIR, f)
        print(f"  {f} ({os.path.getsize(fp)} bytes)")
except Exception as e:
    import traceback
    print(f"[{time.strftime('%X')}] ❌ ERROR:", e)
    traceback.print_exc()
