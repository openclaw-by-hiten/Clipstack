import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
EXPORTS_DIR = BASE_DIR / "exports"
STATIC_DIR = BASE_DIR / "static"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
for folder in [WORKSPACE_DIR, EXPORTS_DIR, STATIC_DIR, CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Hardware & System Settings
MAX_CLIPS_PER_SESSION = 5
SUPPORTED_DURATIONS = [30, 60, 90, 180]  # seconds

# Default Subtitle Colors & Themes
THEMES = {
    "hormozi": {
        "name": "Hormozi Pop",
        "font": "Impact",
        "fontsize": 86,
        "primary_color": "&H00FFFFFF&",     # White
        "highlight_color": "&H0000FFFF&",   # Neon Yellow
        "outline_color": "&H00000000&",     # Black
        "outline": 8,
        "shadow": 3,
        "animation": "bounce"
    },
    "mrbeast": {
        "name": "Beast Pop",
        "font": "Impact",
        "fontsize": 88,
        "primary_color": "&H00FFFFFF&",     # White
        "highlight_color": "&H00FFFF00&",   # Cyan
        "outline_color": "&H00000000&",
        "outline": 9,
        "shadow": 4,
        "animation": "flash"
    },
    "raj_shamani": {
        "name": "Raj Shamani",
        "font": "Arial",
        "fontsize": 72,
        "primary_color": "&H00FFFFFF&",
        "highlight_color": "&H0000A5FF&",   # Orange
        "outline_color": "&H00000000&",
        "outline": 5,
        "shadow": 2,
        "animation": "smooth"
    },
    "karaoke": {
        "name": "Karaoke Sweep",
        "font": "Arial",
        "fontsize": 76,
        "primary_color": "&H00808080&",     # Gray base
        "highlight_color": "&H0000E5FF&",   # Golden Yellow
        "outline_color": "&H00000000&",
        "outline": 6,
        "shadow": 2,
        "animation": "karaoke"
    },
    "clean": {
        "name": "Minimal Clean",
        "font": "Arial",
        "fontsize": 70,
        "primary_color": "&H00FFFFFF&",
        "highlight_color": "&H0000FF00&",   # Lime Green
        "outline_color": "&H00000000&",
        "outline": 4,
        "shadow": 1,
        "animation": "none"
    }
}
