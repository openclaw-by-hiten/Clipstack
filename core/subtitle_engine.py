import os
from core.config import THEMES


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp: H:MM:SS.cc"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    centis = int(round((s - int(s)) * 100))
    if centis == 100:
        centis = 99
    return f"{h}:{m:02d}:{int(s):02d}.{centis:02d}"


def _resolve_ass_font(theme_font: str, custom_font: str = None) -> str:
    if custom_font:
        return custom_font
    if os.name != "nt":
        # On Linux/Colab, prefer Impact/Arial if msttcorefonts installed, else crisp DejaVu Sans
        if theme_font.lower() == "impact" and not os.path.exists("/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"):
            return "DejaVu Sans"
        if theme_font.lower() == "arial" and not os.path.exists("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"):
            return "DejaVu Sans"
    return theme_font


def generate_ass_subtitles(
    words: list,
    clip_start: float,
    clip_end: float,
    output_path: str,
    theme_key: str = "hormozi",
    aspect_ratio: str = "9:16",
    custom_font: str = None,
    margin_v_override: int = None,
    alignment_override: int = None,
) -> str:
    """
    Generates a high-production-value Advanced SubStation Alpha (.ass) subtitle file.
    Adapts resolution, font scale, alignment, and vertical safe margin to 9:16, 1:1, and 16:9
    as well as Smart-Fit / Duo-Split / Face-Track layouts so captions NEVER overlap faces.
    """
    theme = THEMES.get(theme_key, THEMES["hormozi"])
    font_name = _resolve_ass_font(theme["font"], custom_font)
    base_size = theme["fontsize"]

    if aspect_ratio == "9:16":
        res_x, res_y = 1080, 1920
        font_size = base_size
        margin_v = 420
        margin_lr = 56
    elif aspect_ratio == "1:1":
        res_x, res_y = 1080, 1080
        font_size = int(round(base_size * 0.76))
        margin_v = 115
        margin_lr = 52
    else:  # 16:9
        res_x, res_y = 1920, 1080
        font_size = int(round(base_size * 0.80))
        margin_v = 90
        margin_lr = 90

    if margin_v_override is not None:
        margin_v = int(margin_v_override)
    alignment = int(alignment_override) if alignment_override is not None else 2

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{theme['primary_color']},{theme['highlight_color']},{theme['outline_color']},&H80000000,-1,0,0,0,100,100,2,0,1,{theme['outline']},{theme['shadow']},{alignment},{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    rel_words = []
    for w in words:
        if (w["start"] >= (clip_start - 0.1)) and (w["start"] < clip_end):
            rel_words.append({
                "word": w["word"].strip(),
                "start": max(0.0, round(w["start"] - clip_start, 2)),
                "end": max(0.1, round(w["end"] - clip_start, 2)),
            })

    if not rel_words:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
        return output_path

    # Group into natural, rhythmic 2-3 word phrases (4 max on 16:9 landscape)
    max_words_per_phrase = 4 if aspect_ratio == "16:9" else 3
    phrases = []
    curr_phrase = []
    for i, w in enumerate(rel_words):
        curr_phrase.append(w)
        is_last = i == len(rel_words) - 1
        ends_punct = w["word"].endswith((".", "!", "?", ","))
        has_pause = (not is_last) and ((rel_words[i + 1]["start"] - w["end"]) >= 0.35)
        char_len = sum(len(x["word"]) for x in curr_phrase)

        if (
            len(curr_phrase) >= max_words_per_phrase
            or char_len >= (22 if aspect_ratio != "16:9" else 30)
            or (len(curr_phrase) >= 2 and (ends_punct or has_pause))
            or is_last
        ):
            phrases.append(curr_phrase)
            curr_phrase = []

    if curr_phrase:
        phrases.append(curr_phrase)

    events = []
    global_timeline = 0.0

    for p_idx, phrase in enumerate(phrases):
        p_start = max(global_timeline, phrase[0]["start"])
        p_end = max(phrase[-1]["end"], p_start + 0.35)

        if p_idx < len(phrases) - 1:
            next_p_start = phrases[p_idx + 1][0]["start"]
            p_end = min(p_end, next_p_start)
            if 0 < (next_p_start - p_end) <= 0.4:
                p_end = next_p_start

        prev_line_end = p_start
        for k, active_word in enumerate(phrase):
            line_start = max(global_timeline, prev_line_end)
            if k < len(phrase) - 1:
                line_end = max(line_start + 0.05, phrase[k + 1]["start"])
            else:
                line_end = max(line_start + 0.08, p_end)

            line_end = max(line_start + 0.05, line_end)
            prev_line_end = line_end
            global_timeline = line_end

            text_tokens = []
            for j, item in enumerate(phrase):
                word_clean = item["word"].upper()
                if j == k:
                    if theme["animation"] == "bounce":
                        token = (
                            r"{\c"
                            + theme["highlight_color"]
                            + r"\t(0,60,\fscx112\fscy112)\t(60,120,\fscx100\fscy100)}"
                            + word_clean
                            + r"{\c"
                            + theme["primary_color"]
                            + r"}"
                        )
                    elif theme["animation"] == "flash":
                        token = (
                            r"{\c"
                            + theme["highlight_color"]
                            + r"\frz-2}"
                            + word_clean
                            + r"{\frz0\c"
                            + theme["primary_color"]
                            + r"}"
                        )
                    else:
                        token = (
                            r"{\c"
                            + theme["highlight_color"]
                            + r"}"
                            + word_clean
                            + r"{\c"
                            + theme["primary_color"]
                            + r"}"
                        )
                    text_tokens.append(token)
                else:
                    text_tokens.append(word_clean)

            line_text = " ".join(text_tokens)
            w_start_str = format_ass_time(line_start)
            w_end_str = format_ass_time(line_end)
            events.append(f"Dialogue: 0,{w_start_str},{w_end_str},Default,,0,0,0,,{line_text}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(events)

    return output_path
