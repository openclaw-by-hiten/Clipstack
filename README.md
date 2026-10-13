# ⚡ ClipStack — Neural AI Viral Clipper

ClipStack is a 100% local, high-performance AI video clipping studio powered by **Faster-Whisper** and **Meta TRIBE v2 Neural Brain-Attention Scoring**. It transforms long-form videos (YouTube URLs or local files) into viral, high-retention vertical shorts, reels, and TikToks with automated animated subtitles, anti-AI perimeter floating watermarks, 8K AI thumbnail generation, and multi-platform viral copy & publishing.

---

## 🌟 Key Features

### 1. 🧠 Neural Brain-Retention Mining (Meta TRIBE v2)
* **Mathematical Hook Detection**: Analyzes audio-visual engagement curves to isolate moments with the highest probability of holding viewer attention.
* **Smart Non-Overlapping Cuts**: Extracts 1–5 non-overlapping, high-impact clips per session.
* **Duration Selector**: Presets for **30s**, **60s**, **90s**, and **180s** (Extended Shorts).

### 2. 🌊 Anti-AI Perimeter Floating Watermark
* **Intelligent Edge Drifting**: Watermark continuously drifts along the perimeter (top-left ➔ top-right ➔ bottom-right ➔ bottom-left) at safe borders.
* **Zero Viewing Interference**: Keeps the center screen completely unobstructed while making automated AI watermark-removal tools ineffective.

### 3. ✨ 8K Distinct AI Thumbnail Engine
* **Context-Driven Visual Selection**: Generates unique, high-resolution 8K thumbnail posters for every clip rather than repeating the same frame.
* **Brand Logo Overlay**: Automatically embeds your circular channel badge in designated safe zones without covering critical titles or subjects.

### 4. 🎨 Channel Brand Kit & Profile Management
* **Platform-Specific Handles**: Built-in support for YouTube (`@clipstack_A1`), Instagram (`@clip_stacka1`), and X/Twitter (`@clipstack_A1`).
* **Live Customization**: Upload and swap brand logos directly from the UI without modifying source code.

### 5. ✍️ Multi-Platform Viral Copy & Algorithmic Hashtag Hub
* **Platform-Optimized Formulations**:
  * **🔴 YouTube Shorts**: High-CTR curiosity-gap titles, retention descriptions, and 3–5 targeted algorithmic tags.
  * **📸 Instagram Reels**: DM-share triggers, contextual storytelling captions, and explore-page niche SEO tags.
  * **🐦 X / Twitter**: High-velocity posts (< 280 chars) engineered for quote-tweets and replies.
  * **🎵 TikTok**: Sound hooks and FYP tags.
* **1-Click Copy**: Copy complete packages or individual titles, captions, and tags in one click.

### 6. 📂 Quick Workflow Controls & Local File Sync
* **1-Click Path Copy**: Copy absolute Windows file paths (`Video Path`, `Thumb Path`, `Folder Path`) directly to clipboard for drag-and-drop posting.
* **Permanent Local & Web Deletion**: Safely delete clips from disk storage and the website library simultaneously with confirmation protection.
* **Built-in HTML5 Player**: Instant preview with duration and retention badges.

---

## 📁 Project Architecture

```
ClipStack/
├── core/
│   ├── audio_ducking.py        # Background music auto-ducking via sidechain compression
│   ├── branding.py             # Channel identity, logo compositing & placement
│   ├── browser_auth.py         # 2-click interactive browser authentication
│   ├── config.py               # Path configurations & session caps
│   ├── ingestion.py            # yt-dlp & local video stream extraction
│   ├── miner.py                # Candidate segmentation & moment framing
│   ├── neural_scorer.py        # Meta TRIBE v2 engagement scoring
│   ├── orchestrator.py         # End-to-end pipeline coordination & progress streaming
│   ├── renderer.py             # FFmpeg hardware-accelerated 9:16 re-framing & effects
│   ├── social_publisher.py     # Multi-platform viral copy generator & publishers
│   ├── subtitle_engine.py      # Microsecond animated subtitle styling (Hormozi, Beast, etc.)
│   ├── thumbnail_generator.py  # 8K distinct thumbnail engine
│   ├── transcription.py        # Faster-Whisper speech-to-text
│   └── verifier.py             # Audio-visual boundary validation
├── static/
│   └── index.html              # Responsive Web Studio UI (TailwindCSS + Lucide Icons)
├── workspace/
│   ├── ai_thumbnails/          # 8K hyper-realistic base templates
│   └── branding/               # Channel brand configuration & default logo
├── exports/                    # Output directory for rendered clips & thumbnails
├── server.py                   # FastAPI application & WebSocket real-time progress hub
├── requirements.txt            # Python dependencies
├── run_app.bat                 # 1-Click Windows launcher script
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Windows 10 / 11
* Python 3.10+
* [FFmpeg](https://ffmpeg.org/) installed and available in `PATH`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/openclaw-by-hiten/Clipstack.git
   cd Clipstack
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the Studio:**
   Double-click `run_app.bat` or run:
   ```bash
   venv\Scripts\python.exe server.py
   ```
   Open your browser at **http://localhost:8000**.
