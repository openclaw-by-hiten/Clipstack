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


_COPY_VARIANT_COUNTER = {}

_CAPTION_GARBAGE_RE = re.compile(r">>|\[.*?\]|\(.*?\)|#\w+|https?://\S+")

_POWER_HOOK_WORDS = {
    "bad ending", "good ending", "run over", "never", "secret", "impossible", "million",
    "thousand", "caught", "exposed", "broke", "destroyed", "insane", "crazy", "won",
    "lost", "killed", "died", "arrested", "busted", "police", "cops", "heist", "betrayed",
    "truth", "mistake", "wrong", "shocking", "genius", "clutch", "record", "eliminated",
    "survived", "final", "actually", "literally", "nobody", "everyone", "stopped",
}

_WEAK_OPENERS = {
    "i guess we do this", "let's go", "here we go", "okay", "alright", "so yeah",
    "wait a second", "hold on", "check this out", "look at this",
}


def _clean_caption_text(text: str) -> str:
    if not text:
        return ""
    raw = text
    # If auto-captions have speaker turn marker '>>', prefer the richer clause
    if ">>" in raw:
        parts = [re.sub(r"\s+", " ", _CAPTION_GARBAGE_RE.sub(" ", p)).strip(" ,;:-") for p in raw.split(">>")]
        parts = [p for p in parts if len(p.split()) >= 3]
        if parts:
            raw = max(parts, key=lambda s: len(s.split()))
    cleaned = _CAPTION_GARBAGE_RE.sub(" ", raw)
    cleaned = re.sub(r"^(?:you\s+)?(?:but\s+)?don't\s+(?=i'm\b|we're\b|they're\b)", "", cleaned.strip(), flags=re.I)
    cleaned = re.sub(r"\.\.\.+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:;,")
    return cleaned


def _extract_topic_tag_from_title(source_title: str) -> tuple[str, list[str]]:
    """Extracts a clean human-readable topic name and 1-2 dynamic SEO hashtags from source_title."""
    if not source_title:
        return "", []
    cleaned = re.sub(r"\[.*?\]|\(.*?\)|#\w+|\|.*$", "", source_title).strip(" -–—:|")
    low = cleaned.lower()

    # Exact franchise / high-search entity matching first
    known_entities = [
        (["gta 6", "gta6", "gta vi", "grand theft auto vi", "grand theft auto 6", "vice city", "leonida"], "GTA 6", ["#GTA6", "#GTAVI", "#ViceCity"]),
        (["gta 5", "gta5", "gta v", "gta online", "los santos"], "GTA 5", ["#GTA5", "#GTAOnline"]),
        (["minecraft"], "Minecraft", ["#Minecraft", "#MinecraftClips"]),
        (["fortnite"], "Fortnite", ["#Fortnite", "#FortniteClips"]),
        (["call of duty", "warzone", "modern warfare", "black ops"], "Warzone", ["#Warzone", "#CallOfDuty"]),
        (["valorant"], "Valorant", ["#Valorant", "#ValorantClips"]),
        (["cyberpunk 2077", "cyberpunk"], "Cyberpunk 2077", ["#Cyberpunk2077", "#Gaming"]),
        (["red dead", "rdr2"], "Red Dead", ["#RedDeadRedemption2", "#RDR2"]),
        (["elden ring"], "Elden Ring", ["#EldenRing", "#Soulsborne"]),
        (["spider-man", "spiderman"], "Spider-Man", ["#SpiderMan", "#Marvel"]),
        (["roblox"], "Roblox", ["#Roblox", "#RobloxClips"]),
        (["mrbeast", "mr beast"], "MrBeast", ["#MrBeast", "#Challenge"]),
        (["formula 1", "f1"], "F1", ["#F1", "#Formula1"]),
        (["ufc", "mma"], "UFC", ["#UFC", "#MMA"]),
        (["nba"], "NBA", ["#NBA", "#Basketball"]),
        (["nvidia", "openai", "chatgpt", "claude", "gemini"], "AI", ["#AI", "#TechNews"]),
    ]
    for triggers, label, tags in known_entities:
        if any(t in low for t in triggers):
            return label, tags

    # Otherwise extract clean Title-Case entity (avoid verb-led titles like "Lose 150" or "Survive 7")
    skip_words = {
        "the", "and", "for", "with", "this", "that", "how", "why", "what", "when", "official",
        "gameplay", "walkthrough", "playthrough", "episode", "part", "full", "video", "clip",
        "clips", "shorts", "lose", "win", "survive", "make", "get", "stop", "watch", "trying",
        "playing", "building", "doing", "going", "spent", "days", "hours", "together",
    }
    words = [w for w in re.findall(r"[A-Za-z0-9]+", cleaned) if len(w) >= 3 and w.lower() not in skip_words and not w.isdigit()]
    if not words:
        return "", []
    # Only promote to a prefix label if first word looks like a proper brand/subject (not generic)
    first_raw = cleaned.split()[0].lower() if cleaned.split() else ""
    if first_raw in skip_words:
        return "", []
    label = " ".join(words[:2]).title()
    dyn_tag = "#" + "".join(w.capitalize() for w in words[:2])
    return label, ([dyn_tag] if 4 <= len(dyn_tag) <= 22 else [])


def _pick_best_spoken_moments(transcript: str, hook_line: str, payoff_line: str) -> tuple[str, str]:
    """
    Finds the highest-curiosity spoken hook and a distinct secondary payoff line
    from the clip transcript (ignoring bland 2-word fillers or broken fragments).
    """
    candidates = []
    if transcript:
        for s in re.split(r"(?<=[.!?])\s+", transcript):
            c = _clean_caption_text(s)
            words = c.split()
            if 4 <= len(words) <= 18:
                low = c.lower()
                score = 0
                if any(pw in low for pw in _POWER_HOOK_WORDS):
                    score += 5
                if "?" in c or "!" in c:
                    score += 2
                if re.search(r"\$?\d+", c):
                    score += 3
                if low in _WEAK_OPENERS:
                    score -= 6
                if 6 <= len(words) <= 13:
                    score += 2
                candidates.append((score, c))

    clean_hook = _clean_caption_text(hook_line)
    clean_payoff = _clean_caption_text(payoff_line)

    best_hook = clean_hook
    if not best_hook or best_hook.lower() in _WEAK_OPENERS or len(best_hook.split()) < 4:
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_hook = candidates[0][1]

    best_payoff = ""
    if clean_payoff and clean_payoff.lower() != best_hook.lower() and len(clean_payoff.split()) >= 4:
        best_payoff = clean_payoff
    elif candidates:
        for _, c in sorted(candidates, key=lambda x: x[0], reverse=True):
            if c.lower() != best_hook.lower():
                best_payoff = c
                break

    if len(best_hook) > 95:
        best_hook = best_hook[:92].rsplit(" ", 1)[0]
    if len(best_payoff) > 110:
        best_payoff = best_payoff[:107].rsplit(" ", 1)[0] + "..."

    return best_hook, best_payoff


def _extract_clip_intelligence(meta: dict, override_title: str = None) -> dict:
    """
    Deeply analyzes the clip's transcript, hook, payoff, source_title, and stakes
    to build high-CTR, platform-native viral copy with zero generic fluff.
    """
    source_title = (meta.get("source_title") or "").strip()
    raw_title = _clean_caption_text(override_title or meta.get("title") or source_title or "Unbelievable Moment")
    hook_line = _clean_caption_text(meta.get("hook_line") or raw_title)
    payoff_line = _clean_caption_text(meta.get("payoff_line") or "")
    transcript = (meta.get("transcript") or "").strip()

    combined = f"{source_title} {raw_title} {transcript}".lower()
    topic_label, topic_tags = _extract_topic_tag_from_title(f"{source_title} {raw_title}")

    # Extract notable dollar stakes / metrics directly from THIS clip first, then source_title
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

    # High-precision vertical profiles with real high-CTR SEO tags + platform-native debate & share triggers
    niche_profiles = [
        (
            "gta6",
            ["gta 6", "gta6", "gta vi", "grand theft auto", "vice city", "leonida", "rockstar", "gta 5", "gta5", "gta online", "los santos"],
            ["#GTA6", "#GTAVI", "#ViceCity", "#RockstarGames", "#Gaming"],
            ["#gta6", "#gtavi", "#vicecity", "#rockstargames", "#gta6clips", "#gamingreels"],
            ["#gta6", "#gtavi", "#vicecity", "#gta6gameplay", "#gamingclips"],
            [
                "Be honest — are you playing it clean or going straight for the bad ending? 💀👇",
                "1 to 10, how fast would you have gotten a 5-star wanted level here? 🚔👇",
                "Who else is taking off work the day this drops? Drop your honest hype level 👇🔥",
                "Wait till the last second — did you catch what went wrong right there? 😭👇",
            ],
            "Send this to the friend who ALWAYS ruins the mission in GTA 😭👇",
        ),
        (
            "gaming",
            ["game", "gaming", "gamer", "clutch", "rank", "boss", "minecraft", "valorant", "fortnite", "warzone", "elden ring", "cyberpunk", "bad ending", "good ending", "speedrun", "playthrough", "npc", "mission", "ps5", "xbox", "stream"],
            ["#Gaming", "#GamingClips", "#Clutch", "#Gamer", "#VideoGames"],
            ["#gaming", "#gamingclips", "#gamerlife", "#gamingmoments", "#clutchmoment", "#videogames"],
            ["#gaming", "#gamertok", "#gamingclips", "#gamer", "#clutch"],
            [
                "Be honest — would you have choked this moment or clutched it? 🎮👇",
                "Rate this chaos from 1 to 10 in the comments — no lying 💀👇",
                "Wait till the end… how did that actually happen?! 😭👇",
                "Are you going for the safe play or risking it ALL here? 👇🔥",
            ],
            "Send this to your duo who ALWAYS pulls moves like this 🎮😭",
        ),
        (
            "survival",
            ["raft", "ocean", "stranded", "island", "survive", "survival", "jungle", "wilderness", "shark"],
            ["#Survival", "#ExtremeChallenge", "#HighStakes", "#Adventure"],
            ["#survival", "#extremechallenge", "#adventure", "#highstakes", "#viralclips", "#mustwatch"],
            ["#survival", "#challenge", "#highstakes", "#adventure", "#viral"],
            [
                "Be real — how many hours would you actually last out here? 🌊👇",
                "Would you push through or tap out on Day 1? Drop your honest answer 👇",
                "Wait till the twist at the end — could you have survived that?! 😳👇",
                "No phone, no comfort, pure survival. Are you taking this challenge? 👇🔥",
            ],
            "Send this to the friend who thinks they'd survive 30 days on an island 😂🏝️",
        ),
        (
            "fitness",
            ["lbs", "pounds", "weight", "gym", "workout", "calories", "lose", "fat", "muscle", "fitness", "transformation", "deadlift"],
            ["#Transformation", "#FitnessMotivation", "#WeightLossJourney", "#GymMotivation"],
            ["#transformation", "#fitnessmotivation", "#weightlossjourney", "#gymmotivation", "#discipline", "#fitnessreels"],
            ["#transformation", "#gymtok", "#fitness", "#weightloss", "#discipline"],
            [
                "Most people quit in week 2 — could you stay locked in like this? 💪👇",
                "Drop a 🔥 if this just gave you zero excuses to skip today's workout!",
                "Discipline beats motivation every single time. Agree or disagree? 👇",
                "What's the hardest part of a transformation like this — diet or consistency? 👇",
            ],
            "Send this to your gym partner who needs this push today 💪🔥",
        ),
        (
            "challenge",
            ["mrbeast", "challenge", "prize", "win", "won", "eliminated", "contestant", "competition", "vault", "$"],
            ["#Challenge", "#HighStakes", "#ViralChallenge", "#BigWin"],
            ["#challenge", "#highstakes", "#viralchallenge", "#bigwin", "#epicmoments", "#trendingnow"],
            ["#challenge", "#highstakes", "#viral", "#epic", "#win"],
            [
                f"Be honest — would you risk it all{f' for {key_stake}' if key_stake else ''} or walk away safe? 💰👇",
                "My heart rate spiked just watching this… would you have cracked under pressure? 😳👇",
                "99% of people fold right here. Could you actually pull this off? 👇🔥",
                "Take the guaranteed money or risk everything on the final round? Vote below 👇",
            ],
            f"Send this to the friend you'd team up with{f' to split {key_stake}' if key_stake else ' for this challenge'} 💰👇",
        ),
        (
            "motorsport",
            ["f1", "formula 1", "ferrari", "lamborghini", "bugatti", "porsche", "drift", "supercar", "hypercar", "racing", "car"],
            ["#Supercars", "#CarCulture", "#Drift", "#Motorsport"],
            ["#supercars", "#carlovers", "#hypercars", "#drift", "#motorsport", "#carreels"],
            ["#cartok", "#supercars", "#drift", "#racing", "#cars"],
            [
                "Would you trust yourself behind the wheel of this? Be honest 🏎️👇",
                "Rate that sound & control from 1 to 10 — pure adrenaline 🔥👇",
                "Dream car or overrated? Start the debate below 👇💨",
                "Name one car that beats this on the street — I'll wait 👀👇",
            ],
            "Send this to your car-obsessed friend who needs to hear this 🏎️🔥",
        ),
        (
            "sports",
            ["ufc", "mma", "boxing", "knockout", "nba", "basketball", "football", "soccer", "messi", "ronaldo", "cricket", "nfl"],
            ["#Sports", "#Highlights", "#Clutch", "#SportsMoments"],
            ["#sports", "#sportshighlights", "#clutch", "#athlete", "#mentality", "#sportsreels"],
            ["#sportstok", "#highlights", "#clutch", "#sports", "#goat"],
            [
                "Was this pure skill or the luckiest moment of the season? Debate below 🏆👇",
                "Name a more clutch moment than this — I'll wait 🔥👇",
                "1 to 10, where does this rank all-time? Drop your score 👇",
                "Instant goosebumps. Did you see that coming?! 🤯👇",
            ],
            "Send this to the friend who swore this was impossible 🏆🔥",
        ),
        (
            "business",
            ["business", "startup", "founder", "investor", "revenue", "sales", "marketing", "entrepreneur", "ceo", "wealth", "income", "millionaire", "billionaire", "crypto", "investing"],
            ["#BusinessMindset", "#Entrepreneurship", "#WealthBuilding", "#Success"],
            ["#entrepreneur", "#businessmindset", "#wealthbuilding", "#founder", "#moneymindset", "#success"],
            ["#businesstok", "#entrepreneur", "#moneytok", "#wealth", "#mindset"],
            [
                "Hard truth or bad advice? Drop your honest take in the comments 👇",
                "Most people hear this and still do the exact opposite. Do you agree? 💡👇",
                "Save this before you forget it — which part hit hardest for you? 👇",
                "Wish someone told me this 5 years ago. Agree or disagree? 👇🔥",
            ],
            "Send this to someone building something big right now 📈🔥",
        ),
        (
            "tech",
            ["ai", "artificial intelligence", "software", "code", "coding", "developer", "tech", "apple", "google", "nvidia", "gpu", "robot", "tesla", "spacex", "openai"],
            ["#AI", "#TechNews", "#FutureTech", "#Innovation"],
            ["#ai", "#artificialintelligence", "#technews", "#futuretech", "#innovation", "#technology"],
            ["#techtok", "#ai", "#futuretech", "#technews", "#innovation"],
            [
                "Impressive or terrifying? Where do we end up in 2 years with this? 🤖👇",
                "Is this going to replace jobs or create massive opportunities? Debate below 👇",
                "Nobody is ready for how fast this is moving. Thoughts? ⚡👇",
                "Would you actually use this every day or skip it? 👇",
            ],
            "Send this to your tech friend who still thinks AI is just hype 🤖⚡",
        ),
        (
            "podcast",
            ["podcast", "interview", "conversation", "psychology", "habit", "habits", "relationship", "advice", "lesson", "career", "mindset", "truth"],
            ["#PodcastClips", "#Mindset", "#RealTalk", "#LifeAdvice"],
            ["#podcastclips", "#realtalk", "#mindsetshift", "#lifelessons", "#psychology", "#perspective"],
            ["#podcastclips", "#realtalk", "#mindset", "#advice", "#storytime"],
            [
                "Read that again… do you agree with this or think they went too far? 🎙️👇",
                "100% facts or totally wrong? Let's hear the comments 👇",
                "This perspective hits different once you experience it yourself. Thoughts? 👇",
                "Which line hit you the hardest here? Save this for later 🔖👇",
            ],
            "Send this to the one person who needs to hear this today 🎙️👇",
        ),
    ]

    default_profile = (
        "story",
        [],
        ["#MustWatch", "#Storytime", "#PlotTwist", "#ViralClips"],
        ["#mustwatch", "#storytime", "#plottwist", "#viralclips", "#trendingnow", "#unbelievable"],
        ["#storytime", "#plottwist", "#mustwatch", "#viral", "#fyp"],
        [
            "Wait till the end — what would you have actually done right here?! 😳👇",
            "Nobody saw that coming… rate this moment 1 to 10 👇🔥",
            "Be honest — are you siding with them or calling them out? 👇",
            "I had to replay that twice to make sure I heard it right 😭👇",
        ],
        "Send this to someone who won't believe what just happened 👀👇",
    )

    chosen = default_profile
    best_hits = 0
    for prof in niche_profiles:
        hits = sum(
            1 for kw in prof[1]
            if (kw == "$" and "$" in combined)
            or (len(kw) > 1 and (kw in combined if " " in kw else re.search(rf"\b{re.escape(kw)}\b", combined)))
        )
        # Boost specific game/franchise matches over generic challenge/story
        if prof[0] == "gta6" and hits >= 1:
            hits += 5
        if hits > best_hits:
            best_hits = hits
            chosen = prof

    best_hook, best_payoff = _pick_best_spoken_moments(transcript, hook_line, payoff_line)
    clean_headline = best_hook or raw_title or source_title or "You Won't Believe This Moment"

    # Merge dynamic topic tags from source_title right at the front of each platform's tag list
    def _merge_tags(primary: list[str], extra: list[str], limit: int, lower: bool = False) -> list[str]:
        seen = set()
        out = []
        for t in primary + extra:
            norm = t.lower() if lower else t
            key = norm.lower()
            if key not in seen:
                seen.add(key)
                out.append(norm)
            if len(out) >= limit:
                break
        return out

    aspect_ratio = meta.get("aspect_ratio", "9:16")
    is_short = aspect_ratio != "16:9"

    yt_base_tags = (["#Shorts"] if is_short else []) + topic_tags + chosen[2]
    yt_tags_final = _merge_tags(yt_base_tags, [], limit=5, lower=False)
    ig_tags_final = _merge_tags([t.lower() for t in topic_tags], chosen[3], limit=7, lower=True)
    tt_tags_final = _merge_tags([t.lower() for t in topic_tags], chosen[4], limit=5, lower=True)

    return {
        "niche": chosen[0],
        "topic_label": topic_label,
        "aspect_ratio": aspect_ratio,
        "is_short": is_short,
        "headline": clean_headline,
        "raw_title": raw_title,
        "source_title": source_title,
        "hook_line": best_hook,
        "payoff_line": best_payoff,
        "transcript": transcript[:700],
        "key_stake": key_stake,
        "clip_stat": clip_specific_stat,
        "yt_tags": yt_tags_final,
        "ig_tags": ig_tags_final,
        "tt_tags": tt_tags_final,
        "questions": chosen[5],
        "share_cta": chosen[6],
    }


def _try_llm_viral_copy(intel: dict, handles: dict, variant_idx: int) -> dict | None:
    """
    Optional AI copywriter upgrade if GEMINI_API_KEY or OPENAI_API_KEY is configured.
    Produces ultra-tailored, human-sounding viral copy for all 4 platforms.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not gemini_key and not openai_key:
        return None

    angles = [
        "Curiosity-Gap Open Loop (tease the twist without spoiling it)",
        "Debate & Polarizing Question (trigger instant comment arguments)",
        "Relatable POV / Chaos Humor (native TikTok/Reels creator voice)",
        "High-Stakes Action Hook (punchy, urgent, scroll-stopping)",
    ]
    angle = angles[variant_idx % len(angles)]

    prompt = f"""You are an elite viral short-form social media strategist (10M+ views/month across YouTube Shorts, Instagram Reels, TikTok, X).
Write scroll-stopping, human-sounding, high-CTR social copy for THIS EXACT VIDEO CLIP using angle: {angle}.

CLIP DATA:
- Topic/Franchise: {intel['topic_label'] or intel['niche']}
- Original Video Title: {intel['source_title'] or 'N/A'}
- Clip Spoken Hook: {intel['hook_line']}
- Clip Secondary Moment: {intel['payoff_line'] or 'N/A'}
- Clip Transcript Excerpt: {intel['transcript'][:450] or 'N/A'}
- Format: {'Vertical Short/Reel (9:16)' if intel['is_short'] else 'Landscape Video (16:9)'}
- Handles: YouTube={handles['yt']}, Instagram={handles['ig']}, X={handles['x']}, TikTok={handles['tt']}

STRICT RULES:
- Sound 100% like a real top viral creator — NEVER robotic ("high-retention highlights", "daily highlights", etc.).
- Anchor the exact topic (e.g., GTA 6, MrBeast, etc.) naturally so viewers immediately know what the video is about.
- YouTube title: <= 85 chars, high curiosity CTR, include #Shorts at end ONLY if vertical.
- Instagram caption: killer line-1 hook (< 95 chars before "...more"), short punchy spacing, a DM "Send this to..." share trigger, and 5-7 targeted niche hashtags.
- TikTok caption: ultra-punchy native 1-2 line hook + searchable keywords + 4-5 niche tags (<= 190 chars).
- X post_text: sharp, quote-tweet-worthy, <= 250 chars total, max 2 hashtags.
- Every platform gets a unique, debate-sparking pinned_comment.

Return ONLY valid JSON matching this exact schema:
{{
  "youtube": {{"title": "...", "description": "...", "pinned_comment": "...", "hashtags": ["#Tag1", "#Tag2"]}},
  "instagram": {{"hook": "...", "caption": "...", "pinned_comment": "...", "hashtags": ["#tag1", "#tag2"]}},
  "twitter": {{"post_text": "...", "pinned_comment": "...", "hashtags": ["#Tag1", "#Tag2"]}},
  "tiktok": {{"caption": "...", "pinned_comment": "...", "hashtags": ["#tag1", "#tag2"]}}
}}"""

    try:
        raw_text = ""
        if gemini_key:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key.strip()}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.85},
                },
                timeout=15,
            )
            if r.status_code == 200:
                raw_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif openai_key:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key.strip()}"},
                json={
                    "model": "gpt-4o-mini",
                    "response_format": {"type": "json_object"},
                    "temperature": 0.85,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            if r.status_code == 200:
                raw_text = r.json()["choices"][0]["message"]["content"]

        if raw_text:
            parsed = json.loads(raw_text)
            if all(k in parsed for k in ("youtube", "instagram", "twitter", "tiktok")):
                return parsed
    except Exception as e:
        print(f"[SocialCopy] Optional LLM fallback to native viral engine: {e}")
    return None


def _fit_yt_title(core: str, is_short: bool) -> str:
    suffix = " #Shorts" if is_short else ""
    limit = 92 - len(suffix)
    text = re.sub(r"\s+", " ", core).strip()
    if len(text) > limit:
        text = text[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,;:-") + "..."
    return f"{text}{suffix}"


def generate_viral_social_copy(clip_id: str, title: str = None) -> dict:
    """
    Generates scroll-stopping, platform-native, algorithm-optimized viral copy for
    YouTube Shorts, Instagram Reels, TikTok, and X/Twitter.
    Rotates across 4 proven viral frameworks each time Regenerate is clicked.
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

    # Rotate creative angle each time the user clicks "Regenerate" on this clip
    variant_idx = _COPY_VARIANT_COUNTER.get(clean_id, 0)
    _COPY_VARIANT_COUNTER[clean_id] = (variant_idx + 1) % 4

    intel = _extract_clip_intelligence(meta, override_title=title)

    from core.branding import get_brand_config
    brand_cfg = get_brand_config()
    default_handle = brand_cfg.get("channel_handle", "@clipstack_A1")
    handles = {
        "default": default_handle,
        "yt": brand_cfg.get("youtube_handle") or default_handle,
        "ig": brand_cfg.get("instagram_handle") or "@clip_stacka1",
        "x": brand_cfg.get("twitter_handle") or default_handle,
        "tt": brand_cfg.get("tiktok_handle") or default_handle,
    }

    # Try optional LLM copywriter if API key is set; otherwise use our built-in viral engine
    llm_data = _try_llm_viral_copy(intel, handles, variant_idx)
    if llm_data:
        yt = llm_data["youtube"]
        ig = llm_data["instagram"]
        tw = llm_data["twitter"]
        tt = llm_data["tiktok"]
        yt_tags = yt.get("hashtags") or intel["yt_tags"]
        ig_tags = ig.get("hashtags") or intel["ig_tags"]
        x_tags = tw.get("hashtags") or intel["yt_tags"][:2]
        tt_tags = tt.get("hashtags") or intel["tt_tags"]
        return {
            "youtube": {
                "title": yt.get("title", intel["headline"]),
                "description": yt.get("description", ""),
                "pinned_comment": yt.get("pinned_comment", ""),
                "hashtags": yt_tags,
                "hashtags_str": " ".join(yt_tags),
                "handle": handles["yt"],
            },
            "instagram": {
                "hook": ig.get("hook", intel["headline"]),
                "caption": ig.get("caption", ""),
                "pinned_comment": ig.get("pinned_comment", ""),
                "hashtags": ig_tags,
                "hashtags_str": " ".join(ig_tags),
                "handle": handles["ig"],
            },
            "twitter": {
                "post_text": tw.get("post_text", "")[:278],
                "pinned_comment": tw.get("pinned_comment", ""),
                "hashtags": x_tags,
                "hashtags_str": " ".join(x_tags),
                "handle": handles["x"],
            },
            "tiktok": {
                "caption": tt.get("caption", ""),
                "pinned_comment": tt.get("pinned_comment", ""),
                "hashtags": tt_tags,
                "hashtags_str": " ".join(tt_tags),
                "handle": handles["tt"],
            },
            "default_title": yt.get("title", intel["headline"]),
            "default_hashtags": yt_tags,
            "pinned_comment": yt.get("pinned_comment", ""),
            "niche": intel["niche"],
            "channel_handle": handles["default"],
            "youtube_handle": handles["yt"],
            "instagram_handle": handles["ig"],
            "twitter_handle": handles["x"],
            "tiktok_handle": handles["tt"],
        }

    # --- Built-in Zero-Key Viral Growth Engine (4 Rotating Angles) ---
    headline = intel["headline"].rstrip(".!?")
    topic = intel["topic_label"]
    source_title = intel["source_title"]
    payoff = intel["payoff_line"]
    stake = intel["key_stake"]
    is_short = intel["is_short"]
    questions = intel["questions"]
    question = questions[variant_idx % len(questions)]
    alt_question = questions[(variant_idx + 1) % len(questions)]
    share_cta = intel["share_cta"]

    niche = intel["niche"]
    niche_fallback_ref = {
        "gta6": "GTA 6",
        "gaming": "this game",
        "survival": "this survival challenge",
        "fitness": "this transformation",
        "challenge": "this challenge",
        "motorsport": "this build",
        "sports": "this play",
        "business": "business",
        "tech": "AI & tech",
        "podcast": "this conversation",
        "story": "this clip",
    }
    prefix = f"{topic}: " if (topic and topic.lower() not in headline.lower()) else ""
    topic_ref = topic or niche_fallback_ref.get(niche, "this clip")
    short_quote = headline if len(headline) <= 56 else headline[:53].rsplit(" ", 1)[0]
    is_serious_niche = niche in ("fitness", "business", "tech", "podcast")

    # 1. YOUTUBE SHORTS / VIDEO (High-CTR Curiosity Title + SEO Description + Debate Pinned Comment)
    if is_serious_niche:
        yt_title_variants = [
            f'{prefix}"{short_quote}" — Watch Till The End! 🔥',
            f"{prefix}{short_quote} (This Changes Everything) 💡",
            f"{prefix}Nobody Talks About This: {short_quote} 🎯",
            f"{prefix}The Moment Everything Clicked — {short_quote} 🔥",
        ]
    else:
        yt_title_variants = [
            f'{prefix}"{short_quote}" — Wait Till The End! 💀',
            f"{prefix}{short_quote}… Instant Regret?! 😳🔥",
            f"{prefix}Nobody Expected This Moment ({short_quote}) 🤯",
            f"{prefix}The Moment Everything Went Wrong: {short_quote}! 🔥",
        ]
    if stake and stake.lower() not in short_quote.lower():
        yt_title_variants[3] = f"{prefix}{stake} On The Line — {short_quote}! 🤯"

    yt_title = _fit_yt_title(yt_title_variants[variant_idx], is_short)

    if is_serious_niche:
        yt_desc_hooks = [
            f'"{headline}." — this is the part 99% of people miss. 🔥',
            f"Watch closely — this single moment in {topic_ref} hits different. 👀",
            f"Hard truth right here: {headline}. 💡",
            f"Save this and come back to it later — pure gold in {topic_ref}. 🎯",
        ]
    else:
        yt_desc_hooks = [
            f'"{headline}." — and what happened right after had everyone losing it. 💀',
            f"One split-second decision in {topic_ref} changed the entire outcome. Watch till the final 5 seconds. 👀",
            f"You can't make this up… {headline}. 🔥",
            f"Things escalated from 0 to 100 instantly in {topic_ref}. 😳",
        ]
    yt_desc_lines = [yt_desc_hooks[variant_idx]]
    if payoff and payoff.lower() != headline.lower():
        yt_desc_lines.append(f'\n⚡ "{payoff}"')
    if source_title and source_title.lower() not in yt_desc_hooks[variant_idx].lower():
        yt_desc_lines.append(f"\n🎬 Video: {source_title}")
    yt_desc_lines.append(f"\n💬 {question}")
    yt_desc_lines.append(f"🔔 Sub to {handles['yt']} if you want Part 2 + more wild {topic_ref} moments!")
    yt_desc_lines.append("\n" + " ".join(intel["yt_tags"]))
    yt_desc = "\n".join(yt_desc_lines)

    yt_pinned = f"📌 {question}"

    # 2. INSTAGRAM REELS (Scroll-Stopping <95-char Hook + DM Send/Share Trigger + Clean Niche SEO)
    if is_serious_niche:
        ig_hooks = [
            f'{prefix}"{short_quote}" 🔥 Watch till the final second…',
            f"99% of people quit right before this happens in {topic_ref}. 💪",
            f'Read that again: "{short_quote}" 💡',
            f"This moment from {topic_ref} deserves a bookmark. 🔖🔥",
        ]
    else:
        ig_hooks = [
            f'{prefix}"{short_quote}" 💀 Wait for what happens next…',
            f"POV: You tried to play it cool in {topic_ref} for 5 seconds 😭🔥",
            f"Bro really said \"{short_quote}\" and didn't hesitate for a second 😳",
            f"That escalated WAY too fast in {topic_ref}… watch till the end 👀🔥",
        ]
    ig_hook = ig_hooks[variant_idx]
    ig_body = [ig_hook, ""]
    if payoff and payoff.lower() != headline.lower():
        ig_body.append(f'Then dropped: "{payoff}" 🔥' if is_serious_niche else f'Then dropped: "{payoff}" 💀')
        ig_body.append("")
    elif source_title and source_title.lower() not in ig_hook.lower():
        ig_body.append(f"Clip from: {source_title}")
        ig_body.append("")
    ig_body.append(f"💬 {question}")
    ig_body.append(f"📲 {share_cta}")
    ig_body.append(f"➕ Follow {handles['ig']} so you don't miss the next drop.")
    ig_body.append("")
    ig_body.append(" ".join(intel["ig_tags"]))
    ig_caption = "\n".join(ig_body)

    ig_pinned = f"{alt_question} (Top comment gets pinned 📌)"

    # 3. TIKTOK (Ultra-Punchy Native Creator Hook + TikTok Search SEO + Loop-Boosting First Comment)
    if is_serious_niche:
        tt_hooks = [
            f'Wait till the end of {topic_ref} — "{short_quote}" 🔥',
            f'99% of people miss this part: "{short_quote}" 💡',
            f'This moment in {topic_ref} hits different… watch the last 5 seconds 👀🔥',
            f'"{short_quote}" — save this before you scroll past 🔖',
        ]
    else:
        tt_hooks = [
            f'POV: {topic_ref} chaos hits immediately — "{short_quote}" 😭💀',
            f'Wait for the end of this {topic_ref} clip… "{short_quote}" 😳🔥',
            f'He really said "{short_quote}" in {topic_ref} and meant it 💀',
            f'No way this actually happened in {topic_ref} 😭 watch the last 5 seconds 👀',
        ]
    tt_tags_str = " ".join(intel["tt_tags"])
    tt_core = tt_hooks[variant_idx]
    if len(tt_core) + len(tt_tags_str) + 2 > 210:
        avail = max(60, 205 - len(tt_tags_str))
        tt_core = tt_core[:avail].rsplit(" ", 1)[0] + "..."
    tt_caption = f"{tt_core}\n{tt_tags_str}"

    tt_pinned_variants = [
        f"Wait till the last 5 seconds 👀 {question}",
        f"Am I the only one who had to replay that?! 🔥 {alt_question}",
        f"{question} (Be completely honest 👇)",
        f"Zero hesitation on that one 😭 {question}",
    ]
    tt_pinned = tt_pinned_variants[variant_idx]

    # 4. X / TWITTER (Sharp Quote-Tweet-Worthy Punchline, <= 260 chars, Max 2 Clean Hashtags)
    x_tags = [t for t in intel["yt_tags"] if t.lower() != "#shorts"][:2]
    if not x_tags:
        x_tags = ["#Viral", "#Clips"]
    if is_serious_niche:
        x_variants = [
            f'{prefix}"{short_quote}."\n\nWatch till the end — hits different. 🔥\n\n{question}',
            f'Most people overlook this in {topic_ref}:\n\n"{short_quote}"\n\n{alt_question}',
            f'Worth a bookmark from {topic_ref}:\n\n"{short_quote}" 💡\n\nAgree or disagree?',
            f'This 60-second moment in {topic_ref} says it all: "{short_quote}" 🎯\n\n{question}',
        ]
    else:
        x_variants = [
            f'{prefix}"{short_quote}."\n\nZero hesitation. Instant chaos. 💀\n\n{question}',
            f'This {topic_ref} clip had me crying 😭\n\n"{short_quote}"\n\n{alt_question}',
            f'Still thinking about this {topic_ref} moment:\n\n"{short_quote}" 😳\n\nWould you have risked it?',
            f'Everything in {topic_ref} goes sideways the second someone says "{short_quote}" 💀\n\n{question}',
        ]
    x_core = x_variants[variant_idx]
    x_footer = f"\n\n{' '.join(x_tags)}"
    if len(x_core) + len(x_footer) > 274:
        avail = max(80, 270 - len(x_footer))
        x_core = x_core[:avail].rsplit(" ", 1)[0] + "..."
    x_post = x_core + x_footer
    x_pinned = f"Drop your honest take below — full vibe from {source_title or topic_ref} 👇 (via {handles['x']})"

    return {
        "youtube": {
            "title": yt_title,
            "description": yt_desc,
            "pinned_comment": yt_pinned,
            "hashtags": intel["yt_tags"],
            "hashtags_str": " ".join(intel["yt_tags"]),
            "handle": handles["yt"],
        },
        "instagram": {
            "hook": ig_hook,
            "caption": ig_caption,
            "pinned_comment": ig_pinned,
            "hashtags": intel["ig_tags"],
            "hashtags_str": " ".join(intel["ig_tags"]),
            "handle": handles["ig"],
        },
        "twitter": {
            "post_text": x_post,
            "pinned_comment": x_pinned,
            "hashtags": x_tags,
            "hashtags_str": " ".join(x_tags),
            "handle": handles["x"],
        },
        "tiktok": {
            "caption": tt_caption,
            "pinned_comment": tt_pinned,
            "hashtags": intel["tt_tags"],
            "hashtags_str": " ".join(intel["tt_tags"]),
            "handle": handles["tt"],
        },
        "default_title": yt_title,
        "default_hashtags": intel["yt_tags"],
        "pinned_comment": yt_pinned,
        "niche": intel["niche"],
        "channel_handle": handles["default"],
        "youtube_handle": handles["yt"],
        "instagram_handle": handles["ig"],
        "twitter_handle": handles["x"],
        "tiktok_handle": handles["tt"],
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
