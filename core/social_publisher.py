import os
import re
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
        "privacy_default": "public",
    },
    "instagram": {
        "enabled": False,
        "access_token": "",
        "instagram_account_id": "",
    },
    "tiktok": {
        "enabled": False,
        "access_token": "",
        "open_id": "",
    },
    "webhook": {
        "enabled": False,
        "url": "",
        "secret": "",
    },
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


def _extract_clip_intelligence(meta: dict, override_title: str = None) -> dict:
    """
    Analyzes the clip's REAL transcript, hook_line, payoff_line, and source_title
    so titles, captions, hashtags, and pinned comments align 100% with the actual video clip.
    """
    source_title = (meta.get("source_title") or "").strip()
    raw_title = (override_title or meta.get("title") or source_title or "Must-Watch Highlight").strip()
    hook_line = (meta.get("hook_line") or raw_title).strip()
    payoff_line = (meta.get("payoff_line") or "").strip()
    transcript = (meta.get("transcript") or "").strip()

    combined = f"{source_title} {raw_title} {transcript}".lower()

    # Extract notable numbers / dollar stakes / metrics directly from THIS clip first, then source_title
    clip_text_only = f"{raw_title} {transcript}"
    clip_money = re.findall(r"\$\d[\d,]*(?:\s*(?:million|billion|thousand|k|m))?", clip_text_only, re.I)
    clip_metrics = re.findall(
        r"\b\d[\d,]*\s*(?:lbs|pounds|kg|days|hours|minutes|years|percent|%)\b",
        clip_text_only,
        re.I,
    )
    all_money = clip_money or re.findall(r"\$\d[\d,]*(?:\s*(?:million|billion|thousand|k|m))?", source_title, re.I)
    all_metrics = clip_metrics or re.findall(
        r"\b\d[\d,]*\s*(?:lbs|pounds|kg|days|hours|minutes|years|percent|%)\b",
        source_title,
        re.I,
    )
    clip_specific_stat = clip_metrics[0] if clip_metrics else (clip_money[0] if clip_money else "")
    key_stake = all_money[0] if all_money else (all_metrics[0] if all_metrics else "")

    # Detect real content vertical / niche
    niche_profiles = [
        (
            "fitness",
            ["lbs", "pounds", "weight", "gym", "workout", "calories", "lose", "fat", "muscle", "fitness"],
            ["#FitnessMotivation", "#WeightLossJourney", "#Transformation", "#GymLife"],
            ["#fitnessreels", "#transformation", "#weightloss", "#workoutmotivation", "#explorepage"],
            ["#fitness", "#transformation", "#gymtok", "#motivation", "#fyp"],
            "Could you stay disciplined long enough to pull off this transformation?",
        ),
        (
            "business",
            ["business", "startup", "founder", "investor", "revenue", "sales", "marketing", "entrepreneur", "ceo", "wealth", "income"],
            ["#BusinessTips", "#Entrepreneurship", "#Mindset", "#Wealth"],
            ["#businessmindset", "#entrepreneur", "#wealthbuilding", "#podcastclips", "#explorepage"],
            ["#businesstok", "#entrepreneur", "#moneytok", "#podcastclips", "#fyp"],
            "Do you agree with this take, or would you approach it differently?",
        ),
        (
            "tech",
            ["ai", "software", "code", "coding", "developer", "tech", "apple", "google", "gpu", "robot", "algorithm", "app"],
            ["#Tech", "#AI", "#Innovation", "#FutureTech"],
            ["#techreels", "#artificialintelligence", "#codinglife", "#innovation", "#explorepage"],
            ["#techtok", "#ai", "#coding", "#futuretech", "#fyp"],
            "Where do you see this technology heading in the next 12 months?",
        ),
        (
            "gaming",
            ["game", "gaming", "gamer", "clutch", "rank", "boss", "minecraft", "gta", "valorant", "fortnite", "stream"],
            ["#Gaming", "#Clutch", "#Gamer", "#GamingClips"],
            ["#gamingreels", "#gamerlife", "#clutchmoment", "#gamingcommunity", "#explorepage"],
            ["#gaming", "#gamertok", "#clutch", "#gamingclips", "#fyp"],
            "Rate this play from 1 to 10 in the comments!",
        ),
        (
            "podcast",
            ["podcast", "interview", "conversation", "psychology", "habit", "habits", "relationship", "advice", "lesson", "career"],
            ["#PodcastClips", "#LifeAdvice", "#Mindset", "#Perspective"],
            ["#podcastclips", "#deepthoughts", "#mindsetshift", "#lifelessons", "#explorepage"],
            ["#podcastclips", "#realadvice", "#mindset", "#storytime", "#fyp"],
            "What part of this stood out to you the most?",
        ),
        (
            "challenge",
            ["challenge", "prize", "win", "won", "eliminated", "survive", "contestant", "competition", "$"],
            ["#Challenge", "#HighStakes", "#MustWatch", "#Viral"],
            ["#challengeaccepted", "#viralclips", "#highstakes", "#mustwatch", "#explorepage"],
            ["#challenge", "#highstakes", "#viral", "#mustwatch", "#fyp"],
            f"Would you take on this challenge{f' for {key_stake}' if key_stake else ''}?",
        ),
    ]

    chosen = (
        "story",
        [],
        ["#MustWatch", "#Storytime", "#Highlights", "#Viral"],
        ["#reelsvideo", "#storytime", "#mustwatch", "#trendingreels", "#explorepage"],
        ["#storytime", "#mustwatch", "#highlights", "#viral", "#fyp"],
        "What would you have done in this exact moment?",
    )
    best_hits = 0
    for prof in niche_profiles:
        hits = sum(1 for kw in prof[1] if re.search(rf"\b{re.escape(kw)}\b", combined) or (kw == "$" and "$" in combined))
        if hits > best_hits:
            best_hits = hits
            chosen = prof

    # Build a crisp, complete headline (never ending mid-clause with random '...')
    clean_headline = re.sub(r"\.\.\.+$", "", raw_title).strip(" -–—:;,")
    if len(clean_headline) < 12 and source_title:
        clean_headline = source_title

    # Extract a substantive spoken quote from the clip transcript (skipping 1-2 word cheers/interjections)
    quote_snippet = ""
    if transcript:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
        chosen_q = ""
        for s in sents:
            alpha_words = [w for w in re.findall(r"[A-Za-z]+", s) if len(w) >= 2]
            if len(alpha_words) >= 5 and s.lower().strip(".!?") != clean_headline.lower().strip(".!?"):
                chosen_q = s
                break
        if not chosen_q and sents:
            joined_first = " ".join(sents[:2]).strip()
            if len(joined_first.split()) >= 5 and joined_first.lower().strip(".!?") != clean_headline.lower().strip(".!?"):
                chosen_q = joined_first
        if chosen_q:
            if len(chosen_q) > 140:
                chosen_q = " ".join(chosen_q[:135].split()[:-1]) + "..."
            quote_snippet = f'"{chosen_q}"'

    # Customize engagement question specifically to THIS clip's spoken moment & numbers
    engagement_q = chosen[5]
    if clip_specific_stat and key_stake and clip_specific_stat.lower() != key_stake.lower():
        engagement_q = f"{clip_specific_stat} on the line for {key_stake} — could you pull this off? Drop your honest take below!"
    elif clip_specific_stat:
        engagement_q = f"Seeing {clip_specific_stat} in action here — could you stay this locked in? Drop your answer below!"
    elif chosen[0] == "fitness" and key_stake:
        engagement_q = f"Could you commit to this transformation ({key_stake})? Drop your honest answer below!"
    elif key_stake and "?" not in engagement_q:
        engagement_q = f"{engagement_q} ({key_stake})"

    aspect_ratio = meta.get("aspect_ratio", "9:16")
    is_short = aspect_ratio != "16:9"
    yt_tags_final = (["#Shorts"] + chosen[2][:4]) if is_short else chosen[2][:4]

    return {
        "niche": chosen[0],
        "aspect_ratio": aspect_ratio,
        "is_short": is_short,
        "headline": clean_headline,
        "source_title": source_title,
        "hook_line": hook_line,
        "payoff_line": payoff_line,
        "quote": quote_snippet,
        "key_stake": key_stake,
        "yt_tags": yt_tags_final,
        "ig_tags": chosen[3],
        "tt_tags": chosen[4],
        "engagement_question": engagement_q,
    }


def generate_viral_social_copy(clip_id: str, title: str = None) -> dict:
    """
    Generates 100% clip-aligned social titles, descriptions, hashtags, and Suggested Pinned Comments
    grounded in the actual spoken transcript & source video topic (zero generic fake templates).
    """
    clean_id = clip_id.replace(".mp4", "").replace(".json", "")
    meta_path = EXPORTS_DIR / f"{clean_id}.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    intel = _extract_clip_intelligence(meta, override_title=title)
    headline = intel["headline"]
    source_title = intel["source_title"]
    quote = intel["quote"]
    question = intel["engagement_question"]

    from core.branding import get_brand_config
    brand_cfg = get_brand_config()
    default_handle = brand_cfg.get("channel_handle", "@clipstack_A1")
    yt_handle = brand_cfg.get("youtube_handle") or default_handle
    ig_handle = brand_cfg.get("instagram_handle") or "@clip_stacka1"
    x_handle = brand_cfg.get("twitter_handle") or default_handle
    tt_handle = brand_cfg.get("tiktok_handle") or default_handle

    # 1. YouTube Shorts / Video (Aligned Title <= 95 chars + Context Description + Pinned Comment)
    yt_base = headline.rstrip(".!?")
    if intel.get("is_short", True):
        yt_title = f"{yt_base}! 🔥 #Shorts" if len(yt_base) <= 82 else f"{yt_base[:80].rsplit(' ', 1)[0]}... #Shorts"
    else:
        yt_title = f"{yt_base}! 🔥" if len(yt_base) <= 92 else f"{yt_base[:90].rsplit(' ', 1)[0]}..."

    desc_lines = [f"{headline}"]
    if quote and quote.lower().strip('"') != headline.lower():
        desc_lines.append(f"\n🎙️ Key Moment: {quote}")
    if source_title and source_title.lower() not in headline.lower():
        desc_lines.append(f"📺 From: {source_title}")
    desc_lines.append(f"\n💬 {question}")
    desc_lines.append(f"🔔 Subscribe to {yt_handle} for more high-retention highlights!")
    desc_lines.append(" ".join(intel["yt_tags"]))
    yt_desc = "\n".join(desc_lines)
    yt_pinned = f"💬 {question} 👇"

    # 2. Instagram Reels (Storytelling Caption + Niche SEO Tags + Pinned Comment)
    ig_hook = headline if headline.endswith(("!", "?")) else f"{headline} 👀🔥"
    ig_body = [ig_hook]
    if quote and quote.lower().strip('"') != headline.lower():
        ig_body.append(f"\n{quote}")
    if intel["payoff_line"] and intel["payoff_line"].lower() != intel["hook_line"].lower():
        ig_body.append(f"\n⚡ {intel['payoff_line']}")
    ig_body.append(f"\n💬 {question}")
    ig_body.append(f"👉 Follow {ig_handle} for daily highlights!")
    ig_body.append("\n" + " ".join(intel["ig_tags"]))
    ig_caption = "\n".join(ig_body)
    ig_pinned = f"{question} Tag someone who needs to see this! 👇"

    # 3. X / Twitter (Strictly <= 280 chars, crisp insight/hook + thread reply)
    x_tags = [intel["yt_tags"][1] if len(intel["yt_tags"]) > 1 else "#Viral", "#Clips"]
    x_core = f"{headline}\n\n{quote}\n\n{question}" if quote and len(headline) + len(quote) < 170 else f"{headline}\n\n{question}"
    x_footer = f"\n\n{' '.join(x_tags)} {x_handle}"
    if len(x_core) + len(x_footer) > 276:
        avail = max(60, 273 - len(x_footer))
        x_core = x_core[:avail].rsplit(" ", 1)[0] + "..."
    x_post = x_core + x_footer
    x_pinned = f"Full context: {source_title or headline} — what's your honest take?"

    # 4. TikTok (Punchy native hook + niche FYP tags + first comment)
    tt_caption = f"{headline} {' '.join(intel['tt_tags'])}"
    if len(tt_caption) > 200:
        tt_caption = f"{headline[:130].rsplit(' ', 1)[0]}... {' '.join(intel['tt_tags'])}"
    tt_pinned = f"{question}"

    return {
        "youtube": {
            "title": yt_title,
            "description": yt_desc,
            "pinned_comment": yt_pinned,
            "hashtags": intel["yt_tags"],
            "hashtags_str": " ".join(intel["yt_tags"]),
            "handle": yt_handle,
        },
        "instagram": {
            "hook": ig_hook,
            "caption": ig_caption,
            "pinned_comment": ig_pinned,
            "hashtags": intel["ig_tags"],
            "hashtags_str": " ".join(intel["ig_tags"]),
            "handle": ig_handle,
        },
        "twitter": {
            "post_text": x_post,
            "pinned_comment": x_pinned,
            "hashtags": x_tags,
            "hashtags_str": " ".join(x_tags),
            "handle": x_handle,
        },
        "tiktok": {
            "caption": tt_caption,
            "pinned_comment": tt_pinned,
            "hashtags": intel["tt_tags"],
            "hashtags_str": " ".join(intel["tt_tags"]),
            "handle": tt_handle,
        },
        "default_title": headline,
        "default_hashtags": intel["yt_tags"],
        "pinned_comment": yt_pinned,
        "niche": intel["niche"],
        "channel_handle": default_handle,
        "youtube_handle": yt_handle,
        "instagram_handle": ig_handle,
        "twitter_handle": x_handle,
        "tiktok_handle": tt_handle,
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
        server_base_url: str = "http://localhost:8000",
    ) -> dict:
        clean_id = clip_id.replace(".mp4", "").replace(".json", "")
        video_path = EXPORTS_DIR / f"{clean_id}.mp4"
        thumb_path = EXPORTS_DIR / f"{clean_id}_thumb.jpg"

        if not video_path.exists():
            raise FileNotFoundError(f"Clip video {clean_id}.mp4 not found in exports.")

        self.config = load_social_config()
        clean_title = title or f"Clip {clean_id}"
        results = {}

        if "youtube" in platforms:
            results["youtube"] = self._publish_youtube(
                video_path=video_path,
                thumb_path=thumb_path,
                title=clean_title,
                caption=caption,
                privacy=privacy,
                schedule_time=schedule_time,
            )

        if "instagram" in platforms:
            results["instagram"] = self._publish_instagram(
                video_url=f"{server_base_url}/api/stream/{clean_id}.mp4",
                thumb_url=f"{server_base_url}/api/stream/{clean_id}_thumb.jpg" if thumb_path.exists() else None,
                caption=caption or clean_title,
            )

        if "tiktok" in platforms:
            results["tiktok"] = self._publish_tiktok(
                video_path=video_path,
                caption=caption or clean_title,
            )

        if "webhook" in platforms or self.config.get("webhook", {}).get("enabled"):
            results["webhook"] = self._dispatch_webhook(
                clip_id=clean_id,
                video_path=str(video_path),
                thumb_path=str(thumb_path) if thumb_path.exists() else None,
                video_url=f"{server_base_url}/api/stream/{clean_id}.mp4",
                thumb_url=f"{server_base_url}/api/stream/{clean_id}_thumb.jpg" if thumb_path.exists() else None,
                title=clean_title,
                caption=caption,
                hashtags=hashtags or [],
            )

        history_entry = {
            "id": f"pub_{int(time.time())}_{clean_id[:8]}",
            "clip_id": clean_id,
            "title": clean_title,
            "platforms": platforms,
            "results": results,
            "scheduled": schedule_time is not None,
            "schedule_time": schedule_time,
            "timestamp": time.time(),
            "created_at": time.time(),
        }
        append_publish_history(history_entry)

        return {
            "status": "success",
            "clip_id": clean_id,
            "platforms_published": list(results.keys()),
            "results": results,
            "details": results,
            "entry": history_entry,
        }

    def _publish_youtube(self, video_path: Path, thumb_path: Path, title: str, caption: str, privacy: str, schedule_time: str) -> dict:
        yt_cfg = self.config.get("youtube", {})
        refresh_token = yt_cfg.get("refresh_token")
        client_id = yt_cfg.get("client_id")
        client_secret = yt_cfg.get("client_secret")

        clean_title = f"{title} #Shorts" if "#Shorts" not in title and "#shorts" not in title else title

        if refresh_token and client_id and client_secret:
            try:
                token_resp = requests.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=10,
                )
                access_token = token_resp.json().get("access_token")
                if not access_token:
                    raise ValueError("Failed to obtain YouTube OAuth access token from refresh token.")

                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(video_path.stat().st_size),
                }

                metadata = {
                    "snippet": {
                        "title": clean_title[:100],
                        "description": caption or clean_title,
                        "categoryId": "24",
                        "tags": ["Shorts", "Viral", "Highlights"],
                    },
                    "status": {
                        "privacyStatus": privacy if not schedule_time else "private",
                        "selfDeclaredMadeForKids": False,
                    },
                }
                if schedule_time:
                    metadata["status"]["publishAt"] = schedule_time

                init_resp = requests.post(
                    "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                    headers=headers,
                    json=metadata,
                    timeout=15,
                )
                upload_url = init_resp.headers.get("Location")

                with open(video_path, "rb") as vf:
                    upload_resp = requests.put(
                        upload_url,
                        headers={"Content-Type": "video/mp4"},
                        data=vf,
                        timeout=120,
                    )
                yt_video_id = upload_resp.json().get("id")

                return {
                    "status": "published",
                    "platform": "youtube",
                    "video_id": yt_video_id,
                    "url": f"https://youtube.com/shorts/{yt_video_id}",
                    "mode": "LIVE",
                    "message": "Published live to YouTube Shorts",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "platform": "youtube",
                    "error": str(e),
                    "mode": "LIVE",
                    "message": str(e),
                }

        sim_id = f"yt_short_{int(time.time())}"
        return {
            "status": "ready_simulated",
            "platform": "youtube",
            "video_id": sim_id,
            "url": f"https://youtube.com/shorts/{sim_id}",
            "message": "Staged with clip-aligned title & description",
            "mode": "SIMULATION",
        }

    def _publish_instagram(self, video_url: str, thumb_url: str, caption: str) -> dict:
        ig_cfg = self.config.get("instagram", {})
        access_token = ig_cfg.get("access_token")
        account_id = ig_cfg.get("instagram_account_id")

        if access_token and account_id:
            try:
                create_url = f"https://graph.facebook.com/v19.0/{account_id}/media"
                payload = {
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption,
                    "access_token": access_token,
                }
                if thumb_url:
                    payload["cover_url"] = thumb_url

                c_resp = requests.post(create_url, data=payload, timeout=20)
                container_id = c_resp.json().get("id")
                time.sleep(5)
                pub_url = f"https://graph.facebook.com/v19.0/{account_id}/media_publish"
                p_resp = requests.post(pub_url, data={"creation_id": container_id, "access_token": access_token}, timeout=20)
                media_id = p_resp.json().get("id")

                return {
                    "status": "published",
                    "platform": "instagram",
                    "media_id": media_id,
                    "url": f"https://instagram.com/reel/{media_id}",
                    "mode": "LIVE",
                    "message": "Published live to Instagram Reels",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "platform": "instagram",
                    "error": str(e),
                    "mode": "LIVE",
                    "message": str(e),
                }

        sim_id = f"reel_{int(time.time())}"
        return {
            "status": "ready_simulated",
            "platform": "instagram",
            "media_id": sim_id,
            "url": f"https://instagram.com/reel/{sim_id}",
            "message": "Reels payload staged with niche SEO caption",
            "mode": "SIMULATION",
        }

    def _publish_tiktok(self, video_path: Path, caption: str) -> dict:
        sim_id = f"tt_{int(time.time())}"
        return {
            "status": "ready_simulated",
            "platform": "tiktok",
            "video_id": sim_id,
            "url": f"https://tiktok.com/@creator/video/{sim_id}",
            "message": "TikTok Short staged with FYP caption",
            "mode": "SIMULATION",
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
        hashtags: list,
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
            "timestamp": time.time(),
        }

        if url and url.startswith("http"):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                return {
                    "status": "dispatched",
                    "platform": "webhook",
                    "response_code": resp.status_code,
                    "url": url,
                    "mode": "LIVE",
                    "message": f"Webhook HTTP {resp.status_code}",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "platform": "webhook",
                    "error": str(e),
                    "message": str(e),
                }

        return {
            "status": "staged",
            "platform": "webhook",
            "payload": payload,
            "mode": "SIMULATION",
            "message": "Webhook payload ready",
        }
