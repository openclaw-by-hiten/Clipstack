# ⚡ ClipStack — Neural AI Viral Clipper

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-yellow?style=for-the-badge&logo=github)](https://openclaw-by-hiten.github.io/Clipstack/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openclaw-by-hiten/Clipstack/blob/main/ClipStack_Cloud_Engine.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-emerald?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

ClipStack is a high-performance AI video clipping studio powered by **Faster-Whisper** and **Meta TRIBE v2 Neural Brain-Attention Scoring**. It transforms long-form videos (YouTube URLs or local files) into viral, high-retention vertical shorts, reels, and TikToks with automated animated subtitles, anti-AI perimeter floating watermarks, 8K AI thumbnail generation, and multi-platform viral copy & publishing.

🌐 **Live Web Studio:** [https://openclaw-by-hiten.github.io/Clipstack/](https://openclaw-by-hiten.github.io/Clipstack/)  
⚡ **1-Click Free Cloud GPU Engine:** [Open in Google Colab](https://colab.research.google.com/github/openclaw-by-hiten/Clipstack/blob/main/ClipStack_Cloud_Engine.ipynb)

---

## ⚡ 1-Click Free Cloud GPU Engine (Google Colab)

Want to clip long YouTube videos without consuming local PC storage or GPU? Run ClipStack directly in Google Colab on Google's **Free NVIDIA T4 GPU** and **100 GB cloud disk**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openclaw-by-hiten/Clipstack/blob/main/ClipStack_Cloud_Engine.ipynb)

### 3-Step Setup:
1. **Open the Notebook**: Click the **Open in Colab** badge above.
2. **Run All**: Ensure GPU is enabled (**Runtime** > **Change runtime type** > **T4 GPU**), then click **Runtime** > **Run all** (or press `Ctrl + F9`).
3. **Connect to Web App**: Cell 3 will start a secure Cloudflare tunnel and output your live backend URL (e.g. `https://xxxx.trycloudflare.com`). Copy it, open [https://openclaw-by-hiten.github.io/Clipstack/](https://openclaw-by-hiten.github.io/Clipstack/), click **Backend** in the top right header, and paste it!

*(All video downloads, Whisper transcription, and FFmpeg rendering happen completely in the cloud. 0 MB of disk space is used on your personal computer!)*

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

## 🌐 Enabling GitHub Pages (Live on the Web)

The repository includes a pre-configured GitHub Actions workflow (`.github/workflows/deploy-pages.yml`) and root `index.html`.

To activate the live site on GitHub:
1. Open your repository on GitHub: [https://github.com/openclaw-by-hiten/Clipstack](https://github.com/openclaw-by-hiten/Clipstack)
2. Go to **Settings** ➔ **Pages** (in the left sidebar).
3. Under **Build and deployment** > **Source**:
   * Select **GitHub Actions** (recommended), or select **Deploy from a branch** ➔ `main` / `root`.
4. Your live site will immediately be published at:
   👉 **`https://openclaw-by-hiten.github.io/Clipstack/`**

> **Note**: The live GitHub Pages interface includes interactive sample clips, working 8K thumbnail previews, live viral copy generation, and a brand kit. To process custom live videos with Whisper and FFmpeg, run the local backend or connect a remote API URL via the top-right **Backend** button.

---

## 📁 Project Architecture

```
ClipStack/
├── .github/workflows/
│   └── deploy-pages.yml        # Automatic GitHub Pages deployment workflow
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
├── index.html                  # Root entrypoint for GitHub Pages hosting
├── Dockerfile                  # Container definition for Cloud deployments (HuggingFace / Render)
├── server.py                   # FastAPI application & WebSocket real-time progress hub
├── requirements.txt            # Python dependencies
├── run_app.bat                 # 1-Click Windows launcher script
└── README.md                   # Project documentation
```

---

## 🚀 Running Locally

### Prerequisites
* Windows 10 / 11, macOS, or Linux
* Python 3.10+
* [FFmpeg](https://ffmpeg.org/) installed and available in `PATH`

### Quick Start

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

---

## 🐳 Running with Docker / Cloud Deployment

You can deploy ClipStack as a Docker container to **Hugging Face Spaces**, **Render**, or **Google Cloud Run**:

```bash
docker build -t clipstack .
docker run -p 8000:7860 clipstack
```
