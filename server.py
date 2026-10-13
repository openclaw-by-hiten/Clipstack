import os
import json
import time
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import EXPORTS_DIR, WORKSPACE_DIR, STATIC_DIR, MAX_CLIPS_PER_SESSION
from core.orchestrator import ClipPipelineOrchestrator
from core.social_publisher import (
    SocialPublisher,
    load_social_config,
    save_social_config,
    load_publish_history,
    generate_viral_social_copy
)
from core.browser_auth import (
    get_all_browser_statuses,
    start_interactive_login,
    disconnect_platform
)
from core.branding import (
    get_brand_config,
    save_brand_config,
    save_brand_logo,
    get_logo_file_path,
    composite_brand_on_thumbnail
)

app = FastAPI(title="Neural Clip Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = ClipPipelineOrchestrator()
social_publisher = SocialPublisher()

# WebSocket connection manager for live progress
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
loop = asyncio.get_event_loop()

class ProcessRequest(BaseModel):
    youtube_url: Optional[str] = None
    local_file: Optional[str] = None
    target_duration: int = 60
    clip_count: int = Field(default=3, ge=1, le=5)
    aspect_ratio: str = "9:16"
    caption_theme: str = "hormozi"
    watermark: str = "@clipstack_A1"
    floating_watermark: bool = True
    header_title: Optional[str] = None
    enable_vignette: bool = True
    enable_grain: bool = True
    ai_thumbnail: bool = True
    custom_thumbnail_path: Optional[str] = None

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def run_session_worker(params: dict):
    """Background worker executing the clipping pipeline and broadcasting status."""
    def progress_callback(data: dict):
        try:
            asyncio.run(manager.broadcast(data))
        except Exception:
            pass

    try:
        result = orchestrator.process_session(params, progress_callback=progress_callback)
        asyncio.run(manager.broadcast({"type": "complete", "result": result}))
    except Exception as e:
        asyncio.run(manager.broadcast({"type": "error", "error": str(e)}))

@app.post("/api/process")
async def start_process(req: ProcessRequest, background_tasks: BackgroundTasks):
    if not req.youtube_url and not req.local_file:
        raise HTTPException(status_code=400, detail="Provide either a YouTube URL or uploaded file.")

    if req.clip_count > MAX_CLIPS_PER_SESSION:
        req.clip_count = MAX_CLIPS_PER_SESSION

    background_tasks.add_task(run_session_worker, req.dict())
    return {"status": "started", "message": "Pipeline initiated in background."}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    dest_path = WORKSPACE_DIR / f"upload_{file.filename}"
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"status": "uploaded", "file_path": str(dest_path)}

@app.get("/api/clips")
async def list_clips():
    clips = []
    exports_abs = str(EXPORTS_DIR.resolve())
    for meta_file in EXPORTS_DIR.glob("*.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                clean_id = data.get("id", meta_file.stem)
                video_file = EXPORTS_DIR / f"{clean_id}.mp4"
                thumb_file = EXPORTS_DIR / f"{clean_id}_thumb.jpg"
                data["video_path"] = str(video_file.resolve())
                data["thumb_path"] = str(thumb_file.resolve())
                data["folder_path"] = exports_abs
                clips.append(data)
        except Exception:
            continue
    clips.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"clips": clips, "folder_path": exports_abs}

@app.get("/api/stream/{filename}")
async def stream_media(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"
    return FileResponse(path=str(file_path), media_type=media_type)

@app.get("/api/download/{filename}")
async def download_clip(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")

@app.delete("/api/clips/{clip_id}")
async def delete_clip(clip_id: str):
    clean_id = clip_id.replace(".mp4", "").replace(".jpg", "").replace(".json", "")
    deleted_files = []
    
    # 1. Delete from EXPORTS_DIR
    for p in EXPORTS_DIR.iterdir():
        if p.is_file():
            if p.stem == clean_id or p.name.startswith(f"{clean_id}.") or p.name.startswith(f"{clean_id}_"):
                try:
                    p.unlink()
                    deleted_files.append(str(p.name))
                except Exception as e:
                    print(f"Error deleting export file {p}: {e}")
                    
    # 2. Delete any matching temporary files in WORKSPACE_DIR (e.g., subtitles .ass)
    for p in WORKSPACE_DIR.iterdir():
        if p.is_file():
            if p.stem == clean_id or p.name.startswith(f"{clean_id}.") or p.name.startswith(f"{clean_id}_"):
                try:
                    p.unlink()
                    deleted_files.append(f"workspace/{p.name}")
                except Exception as e:
                    print(f"Error deleting workspace file {p}: {e}")
                    
    if not deleted_files:
        raise HTTPException(status_code=404, detail=f"No matching files found for clip '{clean_id}'")
        
    return {
        "status": "success",
        "clip_id": clean_id,
        "deleted_files": deleted_files,
        "message": f"Successfully deleted {len(deleted_files)} files for {clean_id}."
    }

# ==========================================
# Autonomous Social Auto-Publishing Endpoints
# ==========================================
class SocialPublishRequest(BaseModel):
    clip_id: str
    platforms: list[str] = ["youtube", "instagram", "tiktok"]
    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[list[str]] = None
    privacy: str = "public"
    schedule_time: Optional[str] = None

class SocialConfigRequest(BaseModel):
    config: dict

class SocialCopyRequest(BaseModel):
    clip_id: str
    title: Optional[str] = None

@app.post("/api/social/publish")
async def publish_social(req: SocialPublishRequest):
    try:
        res = social_publisher.publish_clip_multi(
            clip_id=req.clip_id,
            platforms=req.platforms,
            title=req.title,
            caption=req.caption,
            hashtags=req.hashtags,
            privacy=req.privacy,
            schedule_time=req.schedule_time
        )
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/config")
async def get_social_config():
    return load_social_config()

@app.post("/api/social/config")
async def update_social_config(req: SocialConfigRequest):
    saved = save_social_config(req.config)
    return {"status": "success", "config": saved}

@app.get("/api/social/history")
async def get_social_history():
    return {"history": load_publish_history()}

@app.post("/api/social/generate-copy")
async def get_social_viral_copy(req: SocialCopyRequest):
    return generate_viral_social_copy(req.clip_id, req.title)

# ==========================================
# 2-Click Browser Direct Connect Endpoints
# (Zero Client Secret / Zero Developer Setup)
# ==========================================
class BrowserAuthRequest(BaseModel):
    platform: str

@app.get("/api/social/browser-status")
async def get_browser_status():
    return get_all_browser_statuses()

@app.post("/api/social/connect-browser")
async def connect_browser(req: BrowserAuthRequest):
    return start_interactive_login(req.platform)

@app.post("/api/social/disconnect-browser")
async def disconnect_browser(req: BrowserAuthRequest):
    return disconnect_platform(req.platform)

# ==========================================
# Brand Identity & Channel Profile Endpoints
# ==========================================
class BrandConfigRequest(BaseModel):
    channel_handle: Optional[str] = None
    youtube_handle: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    tiktok_handle: Optional[str] = None
    channel_name: Optional[str] = None
    embed_logo_on_thumbnail: Optional[bool] = None
    watermark_text: Optional[str] = None
    logo_position: Optional[str] = None
    logo_size: Optional[int] = None

@app.get("/api/branding")
async def get_branding():
    cfg = get_brand_config()
    logo_p = get_logo_file_path()
    return {
        **cfg,
        "has_logo": logo_p is not None and logo_p.exists(),
        "logo_url": "/api/branding/logo" if logo_p and logo_p.exists() else None
    }

@app.post("/api/branding")
async def update_branding(req: BrandConfigRequest):
    updates = {k: v for k, v in req.dict().items() if v is not None}
    if "channel_handle" in updates and not updates.get("watermark_text"):
        updates["watermark_text"] = updates["channel_handle"]
    saved = save_brand_config(updates)
    logo_p = get_logo_file_path()
    return {
        "status": "success",
        "config": {
            **saved,
            "has_logo": logo_p is not None and logo_p.exists(),
            "logo_url": "/api/branding/logo" if logo_p and logo_p.exists() else None
        }
    }

@app.post("/api/branding/logo")
async def upload_branding_logo(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    save_brand_logo(content, file.filename)
    cfg = get_brand_config()
    return {
        "status": "success",
        "message": "Brand logo updated successfully.",
        "logo_url": f"/api/branding/logo?t={int(time.time())}",
        "config": cfg
    }

@app.get("/api/branding/logo")
async def get_branding_logo():
    logo_p = get_logo_file_path()
    if not logo_p or not logo_p.exists():
        raise HTTPException(status_code=404, detail="No brand logo found.")
    ext = logo_p.suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(path=str(logo_p), media_type=media_type)

@app.post("/api/branding/rebrand-clip/{clip_id}")
async def rebrand_existing_clip(clip_id: str):
    thumb_path = EXPORTS_DIR / f"{clip_id}_thumb.jpg"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    composite_brand_on_thumbnail(str(thumb_path))
    return {"status": "success", "thumb_url": f"/api/stream/{clip_id}_thumb.jpg?t={int(time.time())}"}

@app.api_route("/api/locate/{clip_id}", methods=["GET", "POST"])
async def locate_file_in_explorer(clip_id: str):
    import subprocess
    clean_id = clip_id.replace(".mp4", "").replace(".jpg", "").replace(".json", "")
    target = EXPORTS_DIR / f"{clean_id}.mp4"
    thumb = EXPORTS_DIR / f"{clean_id}_thumb.jpg"
    folder = EXPORTS_DIR.resolve()
    
    vid_p = str(target.resolve()) if target.exists() else str((EXPORTS_DIR / f"{clean_id}.mp4").resolve())
    thumb_p = str(thumb.resolve()) if thumb.exists() else str((EXPORTS_DIR / f"{clean_id}_thumb.jpg").resolve())
    folder_p = str(folder)
        
    try:
        if os.name == "nt":
            if target.is_file():
                subprocess.Popen(["explorer.exe", f"/select,{vid_p}"])
            else:
                subprocess.Popen(["explorer.exe", folder_p])
        else:
            subprocess.Popen(["xdg-open", folder_p])
    except Exception:
        pass

    return {
        "status": "success",
        "clip_id": clean_id,
        "video_path": vid_p,
        "thumb_path": thumb_p,
        "folder_path": folder_p
    }

@app.api_route("/api/locate-folder", methods=["GET", "POST"])
async def locate_exports_folder():
    folder_p = str(EXPORTS_DIR.resolve())
    try:
        if os.name == "nt":
            import subprocess
            subprocess.Popen(["explorer.exe", folder_p])
    except Exception:
        pass
    return {
        "status": "success",
        "folder_path": folder_p
    }

# ==========================================
# 2-Click Direct Social Connect Bridge & Popup
# ==========================================
class DirectLinkRequest(BaseModel):
    platform: str
    account_name: str
    handle: Optional[str] = None
    privacy: Optional[str] = "public"

@app.post("/api/social/link-direct")
async def link_social_direct(req: DirectLinkRequest):
    import time
    from core.browser_auth import get_status_file, PROFILES_DIR
    
    status_file = get_status_file(req.platform)
    profile_dir = PROFILES_DIR / req.platform
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    clean_name = req.account_name.strip()
    if req.handle and not clean_name.startswith("@"):
        clean_name = f"{clean_name} ({req.handle})"
    
    record = {
        "platform": req.platform,
        "connected": True,
        "connecting": False,
        "account_name": clean_name,
        "privacy": req.privacy,
        "connected_at": time.time(),
        "message": f"Successfully connected as {clean_name}!"
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return {"status": "connected", "record": record}

@app.get("/connect/{platform}", response_class=HTMLResponse)
async def connect_page(platform: str):
    p = platform.lower()
    config_map = {
        "youtube": {
            "name": "YouTube Shorts",
            "icon": "▶",
            "badge_color": "bg-red-500/20 text-red-400 border-red-500/30",
            "btn_gradient": "from-red-600 via-rose-600 to-amber-600",
            "placeholder": "e.g. @HitenClips or Channel Name",
            "official_url": "https://accounts.google.com/ServiceLogin?service=youtube&continue=https%3A%2F%2Fstudio.youtube.com%2F",
            "official_label": "Open YouTube Studio in New Tab (Verify 2FA) ↗"
        },
        "instagram": {
            "name": "Instagram Reels",
            "icon": "📸",
            "badge_color": "bg-pink-500/20 text-pink-400 border-pink-500/30",
            "btn_gradient": "from-pink-600 via-purple-600 to-indigo-600",
            "placeholder": "e.g. @autoclip or your_handle",
            "official_url": "https://www.instagram.com/accounts/login/",
            "official_label": "Open Instagram in New Tab (Verify 2FA) ↗"
        },
        "tiktok": {
            "name": "TikTok",
            "icon": "🎵",
            "badge_color": "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
            "btn_gradient": "from-cyan-500 via-teal-500 to-blue-600",
            "placeholder": "e.g. @tiktok_creator",
            "official_url": "https://www.tiktok.com/login",
            "official_label": "Open TikTok in New Tab (Verify 2FA) ↗"
        }
    }
    cfg = config_map.get(p, config_map["youtube"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Connect {cfg['name']} — Neural Clip Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    body {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #0c0d0e;
    }}
  </style>
</head>
<body class="text-neutral-100 min-h-screen flex items-center justify-center p-4 selection:bg-yellow-400 selection:text-black">
  <div class="max-w-md w-full bg-neutral-900 border border-neutral-800 rounded-3xl p-6 shadow-2xl space-y-6">
    
    <!-- Header -->
    <div class="flex items-center gap-3.5 pb-4 border-b border-neutral-800/80">
      <div class="w-12 h-12 rounded-2xl bg-neutral-950 border border-neutral-800 flex items-center justify-center text-2xl shadow-inner shrink-0">
        {cfg['icon']}
      </div>
      <div>
        <h1 class="text-base font-extrabold text-white flex items-center gap-2">
          Connect {cfg['name']}
          <span class="text-[10px] px-2 py-0.5 rounded-full border {cfg['badge_color']} font-mono font-bold">2-Click Link</span>
        </h1>
        <p class="text-xs text-neutral-400">Direct account authorization for 1-Click Publishing</p>
      </div>
    </div>

    <!-- Form -->
    <form id="connectForm" class="space-y-4">
      <div>
        <label class="block text-[11px] font-bold uppercase tracking-wider text-neutral-400 mb-1.5">
          Account Handle / Channel Name
        </label>
        <input 
          type="text" 
          id="accountName" 
          placeholder="{cfg['placeholder']}" 
          required 
          autofocus
          class="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-yellow-400 transition-colors"
        />
        <p class="text-[10px] text-neutral-500 mt-1">Enter your creator handle or channel title as shown on your profile.</p>
      </div>

      <div>
        <label class="block text-[11px] font-bold uppercase tracking-wider text-neutral-400 mb-1.5">
          Default Publishing Visibility
        </label>
        <select id="privacy" class="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-yellow-400">
          <option value="public" selected>Public (Instant Viral Drop)</option>
          <option value="unlisted">Unlisted (Private Review)</option>
          <option value="private">Private (Draft Mode)</option>
        </select>
      </div>

      <div class="pt-2">
        <button 
          type="submit" 
          id="submitBtn"
          class="w-full bg-gradient-to-r {cfg['btn_gradient']} text-white font-extrabold text-sm py-3.5 rounded-xl shadow-lg transition-all active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer"
        >
          🚀 Authorize & Connect {cfg['name']}
        </button>
      </div>
    </form>

    <!-- 2FA & Web Sign-in Section -->
    <div class="pt-3 border-t border-neutral-800/80 space-y-2.5 text-center">
      <div class="text-[11px] text-neutral-500">Need to check credentials or complete 2FA in a tab?</div>
      <a 
        href="{cfg['official_url']}" 
        target="_blank" 
        class="inline-flex items-center gap-1.5 text-xs text-neutral-300 hover:text-white bg-neutral-950 border border-neutral-800 px-3.5 py-2 rounded-xl transition-colors"
      >
        {cfg['official_label']}
      </a>
    </div>

    <!-- Status feedback -->
    <div id="statusNotice" class="hidden p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold text-center space-y-1">
      <div class="text-sm">✓ Connected Successfully!</div>
      <div class="text-[11px] text-neutral-400">Closing window and updating dashboard...</div>
    </div>

  </div>

  <script>
    const platform = '{p}';
    document.getElementById('connectForm').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const accountName = document.getElementById('accountName').value.trim();
      const privacy = document.getElementById('privacy').value;
      if (!accountName) return;

      const btn = document.getElementById('submitBtn');
      btn.disabled = true;
      btn.innerHTML = 'Connecting...';

      try {{
        const res = await fetch('/api/social/link-direct', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ platform, account_name: accountName, privacy }})
        }});
        const data = await res.json();
        
        document.getElementById('connectForm').classList.add('hidden');
        const notice = document.getElementById('statusNotice');
        notice.classList.remove('hidden');

        // Notify parent window (http://localhost:8000)
        if (window.opener) {{
          window.opener.postMessage({{
            type: 'SOCIAL_CONNECTED',
            platform: platform,
            account: accountName
          }}, '*');
        }}

        setTimeout(() => {{
          window.close();
        }}, 1000);

      }} catch (err) {{
        alert('Failed to connect: ' + err.message);
        btn.disabled = false;
        btn.innerHTML = '🚀 Authorize & Connect {cfg['name']}';
      }}
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)

# Mount workspace assets and static frontend
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR)), name="workspace")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[Server] Starting Neural Clip Studio on http://0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
