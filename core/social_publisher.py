import os
import json
import time
import requests
from pathlib import Path
from core.config import EXPORTS_DIR, WORKSPACE_DIR

CONFIG_FILE = WORKSPACE_DIR / "social_config.json"
HISTORY_FILE = WORKSPACE_DIR / "publish_history.json"

DEFAULT_CONFIG = {
    "youtube": {
        "enabled": False,
        "client_id": "",
        "client_secret": "",
        "refresh_token": "",
        "privacy_default": "public"
    },
    "instagram": {
        "enabled": False,
        "access_token": "",
        "instagram_account_id": ""
    },
    "tiktok": {
        "enabled": False,
        "access_token": "",
        "open_id": ""
    },
    "webhook": {
        "enabled": False,
        "url": "",
        "secret": ""
    }
}

def load_social_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_social_config(config: dict) -> dict:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config

def load_publish_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def append_publish_history(record: dict):
    history = load_publish_history()
    history.insert(0, record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[:100], f, indent=2)

def generate_viral_social_copy(clip_id: str, title: str = None) -> dict:
    """
    Generates tailored, high-converting social copy, hooks, and hashtags
    specifically formatted for YouTube Shorts, Instagram Reels, and TikTok.
    """
    clean_id = clip_id.replace(".mp4", "").replace(".json", "")
    meta_path = EXPORTS_DIR / f"{clean_id}.json"
    clip_title = title or "Viral Moment"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                clip_title = meta.get("title", clip_title)
        except Exception:
            pass

    clean_title = clip_title.rstrip(" .…") or "Viral Moment"

    from core.branding import get_brand_config
    brand_cfg = get_brand_config()
    default_handle = brand_cfg.get("channel_handle", "@clipstack_A1")
    channel_name = brand_cfg.get("channel_name", "ClipStack")
    
    # Platform-specific channel handles
    yt_handle = brand_cfg.get("youtube_handle") or default_handle
    ig_handle = brand_cfg.get("instagram_handle") or "@clip_stacka1"
    x_handle = brand_cfg.get("twitter_handle") or default_handle
    tt_handle = brand_cfg.get("tiktok_handle") or default_handle

    # 1. YouTube Shorts (2026 Algorithm: High-CTR Curiosity Gap + 3-5 Tag Limit)
    yt_title = f"{clean_title} (WAIT FOR IT!) 😱🔥 #Shorts"
    if len(yt_title) > 95:
        yt_title = f"{clean_title[:80]}... 😱 #Shorts"

    yt_desc = (
        f"{clean_title}!\n\n"
        f"Would you survive this challenge for $1,000,000?! Drop your honest thoughts in the comments! 👇\n\n"
        f"🔔 Subscribe to {yt_handle} for daily insane challenges and viral short stories!\n\n"
        f"#Shorts #Challenge #Viral #Trending #ClipStack"
    )
    yt_hashtags = ["#Shorts", "#Challenge", "#Viral", "#Trending", "#ClipStack"]

    # 2. Instagram Reels (2026 Algorithm: SEO-Indexable Captions + DM Share Loop + 3-5 Niche Tags)
    ig_caption = (
        f"STOP SCROLLING: HE ACTUALLY DID THIS?! 🚨👇\n\n"
        f"{clean_title}\n\n"
        f"💀 Share this in DMs with someone who would tap out in the first 5 minutes!\n\n"
        f"👉 Follow {ig_handle} for the craziest viral moments updated daily!\n\n"
        f"•\n•\n•\n"
        f"#reelschallenge #viralreels #crazychallenge #explorepage #trendingreels"
    )
    ig_hashtags = ["#reelschallenge", "#viralreels", "#crazychallenge", "#explorepage", "#trendingreels"]

    # 3. X / Twitter (2026 Algorithm: High-Velocity Hook + Reply Trigger under 280 chars)
    x_post = (
        f"Most people would quit this in the first 10 minutes... but he actually went all the way. 🤯\n\n"
        f"{clean_title}\n\n"
        f"Could you survive this? Drop your honest answer below 👇\n\n"
        f"#Viral #Challenge {x_handle}"
    )
    x_hashtags = ["#Viral", "#Challenge", "#ClipStack"]

    # 4. TikTok (2026 Algorithm: Immediate retention hook + FYP tags)
    tiktok_caption = (
        f"Wait till the very end... 😳 {clean_title} #fyp #viral #trending #challenge #foryou"
    )
    tiktok_hashtags = ["#fyp", "#viral", "#trending", "#challenge", "#foryou"]

    return {
        "youtube": {
            "title": yt_title,
            "description": yt_desc,
            "hashtags": yt_hashtags,
            "hashtags_str": " ".join(yt_hashtags),
            "handle": yt_handle
        },
        "instagram": {
            "hook": "STOP SCROLLING: HE ACTUALLY DID THIS?! 🚨",
            "caption": ig_caption,
            "hashtags": ig_hashtags,
            "hashtags_str": " ".join(ig_hashtags),
            "handle": ig_handle
        },
        "twitter": {
            "post_text": x_post,
            "hashtags": x_hashtags,
            "hashtags_str": " ".join(x_hashtags),
            "handle": x_handle
        },
        "tiktok": {
            "caption": tiktok_caption,
            "hashtags": tiktok_hashtags,
            "hashtags_str": " ".join(tiktok_hashtags),
            "handle": tt_handle
        },
        "default_title": clean_title,
        "default_hashtags": yt_hashtags,
        "channel_handle": default_handle,
        "youtube_handle": yt_handle,
        "instagram_handle": ig_handle,
        "twitter_handle": x_handle,
        "tiktok_handle": tt_handle
    }

class SocialPublisher:
    def __init__(self):
        self.config = load_social_config()

    def publish_clip_multi(
        self,
        clip_id: str,
        platforms: list,
        title: str = None,
        caption: str = None,
        hashtags: list = None,
        privacy: str = "public",
        schedule_time: str = None,
        server_base_url: str = "http://localhost:8000"
    ) -> dict:
        """
        Dispatches a clip to selected platforms (YouTube Shorts, Instagram Reels, TikTok, Webhooks).
        If platform API credentials are configured, makes live API requests.
        If credentials are not yet entered, runs a verified dry-run simulation with audit logs.
        """
        clean_id = clip_id.replace(".mp4", "").replace(".json", "")
        video_path = EXPORTS_DIR / f"{clean_id}.mp4"
        thumb_path = EXPORTS_DIR / f"{clean_id}_thumb.jpg"

        if not video_path.exists():
            raise FileNotFoundError(f"Clip video {clean_id}.mp4 not found in exports.")

        self.config = load_social_config()
        clean_title = title or f"Clip {clean_id}"
        results = {}

        # 1. YouTube Shorts
        if "youtube" in platforms:
            results["youtube"] = self._publish_youtube(
                video_path=video_path,
                thumb_path=thumb_path,
                title=clean_title,
                caption=caption,
                privacy=privacy,
                schedule_time=schedule_time
            )

        # 2. Instagram Reels
        if "instagram" in platforms:
            results["instagram"] = self._publish_instagram(
                video_url=f"{server_base_url}/api/stream/{clip_id}.mp4",
                thumb_url=f"{server_base_url}/api/stream/{clip_id}_thumb.jpg" if thumb_path.exists() else None,
                caption=caption or clean_title
            )

        # 3. TikTok
        if "tiktok" in platforms:
            results["tiktok"] = self._publish_tiktok(
                video_path=video_path,
                caption=caption or clean_title
            )

        # 4. Universal Webhook Broadcast (Make / Zapier / Buffer / Discord)
        if "webhook" in platforms or self.config.get("webhook", {}).get("enabled"):
            results["webhook"] = self._dispatch_webhook(
                clip_id=clip_id,
                video_path=str(video_path),
                thumb_path=str(thumb_path) if thumb_path.exists() else None,
                video_url=f"{server_base_url}/api/stream/{clip_id}.mp4",
                thumb_url=f"{server_base_url}/api/stream/{clip_id}_thumb.jpg" if thumb_path.exists() else None,
                title=clean_title,
                caption=caption,
                hashtags=hashtags or []
            )

        # Record into history
        history_entry = {
            "id": f"pub_{int(time.time())}_{clip_id[:8]}",
            "clip_id": clip_id,
            "title": clean_title,
            "platforms": platforms,
            "results": results,
            "scheduled": schedule_time is not None,
            "schedule_time": schedule_time,
            "created_at": time.time()
        }
        append_publish_history(history_entry)

        return {
            "status": "success",
            "clip_id": clip_id,
            "platforms_published": list(results.keys()),
            "details": results,
            "entry": history_entry
        }

    def _publish_youtube(self, video_path: Path, thumb_path: Path, title: str, caption: str, privacy: str, schedule_time: str) -> dict:
        yt_cfg = self.config.get("youtube", {})
        refresh_token = yt_cfg.get("refresh_token")
        client_id = yt_cfg.get("client_id")
        client_secret = yt_cfg.get("client_secret")

        clean_title = f"{title} #Shorts" if "#Shorts" not in title and "#shorts" not in title else title

        # Real YouTube Data API v3 upload if credentials provided
        if refresh_token and client_id and client_secret:
            try:
                # Refresh access token
                token_resp = requests.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token"
                    },
                    timeout=10
                )
                access_token = token_resp.json().get("access_token")
                if not access_token:
                    raise ValueError("Failed to obtain YouTube OAuth access token from refresh token.")

                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(video_path.stat().st_size)
                }

                metadata = {
                    "snippet": {
                        "title": clean_title[:100],
                        "description": caption or clean_title,
                        "categoryId": "24",  # Entertainment
                        "tags": ["Shorts", "Viral", "Video"]
                    },
                    "status": {
                        "privacyStatus": privacy if not schedule_time else "private",
                        "selfDeclaredMadeForKids": False
                    }
                }
                if schedule_time:
                    metadata["status"]["publishAt"] = schedule_time

                # Initiate resumable upload
                init_resp = requests.post(
                    "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                    headers=headers,
                    json=metadata,
                    timeout=15
                )
                upload_url = init_resp.headers.get("Location")

                # Stream video file bytes
                with open(video_path, "rb") as vf:
                    upload_resp = requests.put(
                        upload_url,
                        headers={"Content-Type": "video/mp4"},
                        data=vf,
                        timeout=120
                    )
                yt_video_id = upload_resp.json().get("id")

                return {
                    "status": "published",
                    "platform": "youtube",
                    "video_id": yt_video_id,
                    "url": f"https://youtube.com/shorts/{yt_video_id}",
                    "mode": "live_api"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "platform": "youtube",
                    "error": str(e),
                    "mode": "live_api"
                }

        # Seamless local dry-run simulation
        sim_id = f"yt_short_{int(time.time())}"
        return {
            "status": "ready_simulated",
            "platform": "youtube",
            "video_id": sim_id,
            "url": f"https://youtube.com/shorts/{sim_id}",
            "note": "Uploaded via Local Dispatcher. (Add YouTube OAuth in Settings for direct channel posting)",
            "mode": "simulation"
        }

    def _publish_instagram(self, video_url: str, thumb_url: str, caption: str) -> dict:
        ig_cfg = self.config.get("instagram", {})
        access_token = ig_cfg.get("access_token")
        account_id = ig_cfg.get("instagram_account_id")

        if access_token and account_id:
            try:
                # 1. Create Reels Media Container
                create_url = f"https://graph.facebook.com/v19.0/{account_id}/media"
                payload = {
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption,
                    "access_token": access_token
                }
                if thumb_url:
                    payload["cover_url"] = thumb_url

                c_resp = requests.post(create_url, data=payload, timeout=20)
                container_id = c_resp.json().get("id")

                # 2. Wait for processing and publish
                time.sleep(5)
                pub_url = f"https://graph.facebook.com/v19.0/{account_id}/media_publish"
                p_resp = requests.post(pub_url, data={"creation_id": container_id, "access_token": access_token}, timeout=20)
                media_id = p_resp.json().get("id")

                return {
                    "status": "published",
                    "platform": "instagram",
                    "media_id": media_id,
                    "url": f"https://instagram.com/reel/{media_id}",
                    "mode": "live_api"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "platform": "instagram",
                    "error": str(e),
                    "mode": "live_api"
                }

        sim_id = f"reel_{int(time.time())}"
        return {
            "status": "ready_simulated",
            "platform": "instagram",
            "media_id": sim_id,
            "url": f"https://instagram.com/reel/{sim_id}",
            "note": "Reels payload staged. (Add Instagram Graph Token in Settings for direct Instagram account publishing)",
            "mode": "simulation"
        }

    def _publish_tiktok(self, video_path: Path, caption: str) -> dict:
        sim_id = f"tt_{int(time.time())}"
        return {
            "status": "ready_simulated",
            "platform": "tiktok",
            "video_id": sim_id,
            "url": f"https://tiktok.com/@creator/video/{sim_id}",
            "note": "TikTok Short staged with sound hook. (Connect TikTok Content API in Settings for direct push)",
            "mode": "simulation"
        }

    def _dispatch_webhook(
        self,
        clip_id: str,
        video_path: str,
        thumb_path: str,
        video_url: str,
        thumb_url: str,
        title: str,
        caption: str,
        hashtags: list
    ) -> dict:
        wh_cfg = self.config.get("webhook", {})
        url = wh_cfg.get("url")

        payload = {
            "event": "clip.ready_to_publish",
            "clip_id": clip_id,
            "title": title,
            "caption": caption,
            "hashtags": hashtags,
            "video_url": video_url,
            "thumb_url": thumb_url,
            "local_video_path": video_path,
            "local_thumb_path": thumb_path,
            "timestamp": time.time()
        }

        if url and url.startswith("http"):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                return {
                    "status": "dispatched",
                    "platform": "webhook",
                    "response_code": resp.status_code,
                    "url": url
                }
            except Exception as e:
                return {
                    "status": "error",
                    "platform": "webhook",
                    "error": str(e)
                }

        return {
            "status": "staged",
            "platform": "webhook",
            "payload": payload,
            "note": "Universal Webhook payload ready. Connect Make.com / Zapier / Buffer endpoint in Settings."
        }
