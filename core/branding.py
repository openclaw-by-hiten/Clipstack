import os
import json
import shutil
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter, ImageOps

BRANDING_DIR = Path(__file__).resolve().parent.parent / "workspace" / "branding"
BRANDING_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = BRANDING_DIR / "brand_config.json"
DEFAULT_LOGO_PATH = BRANDING_DIR / "logo.png"

DEFAULT_CONFIG = {
    "channel_handle": "@clipstack_A1",
    "youtube_handle": "@clipstack_A1",
    "instagram_handle": "@clip_stacka1",
    "twitter_handle": "@clipstack_A1",
    "tiktok_handle": "@clipstack_A1",
    "channel_name": "ClipStack",
    "logo_filename": "logo.png",
    "embed_logo_on_thumbnail": True,
    "logo_position": "bottom_right",
    "logo_size": 125,
    "watermark_text": "@clipstack_A1"
}

def get_brand_config() -> dict:
    """Reads brand configuration from disk, creating default if missing."""
    if not CONFIG_FILE.exists():
        save_brand_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure required keys exist
            merged = {**DEFAULT_CONFIG, **data}
            return merged
    except Exception:
        return dict(DEFAULT_CONFIG)

def save_brand_config(updates: dict) -> dict:
    """Updates and persists brand configuration."""
    current = get_brand_config() if CONFIG_FILE.exists() else dict(DEFAULT_CONFIG)
    current.update(updates)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current

def save_brand_logo(file_bytes: bytes, original_filename: str) -> str:
    """Saves a new uploaded logo and updates configuration."""
    ext = Path(original_filename).suffix.lower() or ".png"
    target_name = f"logo{ext}"
    target_path = BRANDING_DIR / target_name

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    # If not logo.png, also make a canonical copy as logo.png
    if target_name != "logo.png":
        try:
            im = Image.open(target_path)
            im.save(DEFAULT_LOGO_PATH, "PNG")
        except Exception:
            shutil.copy2(target_path, DEFAULT_LOGO_PATH)

    save_brand_config({
        "logo_filename": "logo.png"
    })
    return str(DEFAULT_LOGO_PATH)

def get_logo_file_path() -> Optional[Path]:
    """Returns absolute path to the active brand logo file if it exists."""
    cfg = get_brand_config()
    specific = BRANDING_DIR / cfg.get("logo_filename", "logo.png")
    if specific.exists():
        return specific
    if DEFAULT_LOGO_PATH.exists():
        return DEFAULT_LOGO_PATH
    return None

def create_circular_logo_badge(logo_img: Image.Image, target_size: int = 210) -> Image.Image:
    """Generates a high-finish circular badge with cyan glow rim and drop shadow."""
    logo_img = logo_img.convert("RGBA")
    logo_img = ImageOps.fit(logo_img, (target_size, target_size), Image.Resampling.LANCZOS)
    
    # Circular mask
    mask = Image.new("L", (target_size, target_size), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((0, 0, target_size, target_size), fill=255)
    
    circular_logo = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    circular_logo.paste(logo_img, (0, 0), mask=mask)
    
    # Outer canvas with padding for glow & shadow
    pad = 12
    badge_size = target_size + pad * 2
    badge = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))
    
    # Outer drop shadow
    shadow_mask = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_mask)
    s_draw.ellipse((pad - 2, pad - 2, pad + target_size + 2, pad + target_size + 2), fill=(0, 0, 0, 175))
    shadow_blurred = shadow_mask.filter(ImageFilter.GaussianBlur(6))
    badge = Image.alpha_composite(badge, shadow_blurred)
    
    # Cyan/white rim highlight
    b_draw = ImageDraw.Draw(badge)
    border_bbox = (pad - 3, pad - 3, pad + target_size + 3, pad + target_size + 3)
    b_draw.ellipse(border_bbox, outline=(0, 220, 255, 235), width=4)
    
    # Place circular logo
    badge.paste(circular_logo, (pad, pad), mask=circular_logo)
    return badge

def composite_brand_on_thumbnail(base_thumb_path: str, output_path: str = None) -> str:
    """Composites the active brand logo badge onto a thumbnail if branding is enabled."""
    cfg = get_brand_config()
    if not cfg.get("embed_logo_on_thumbnail", True):
        return base_thumb_path

    logo_path = get_logo_file_path()
    if not logo_path or not os.path.exists(base_thumb_path):
        return base_thumb_path

    if output_path is None:
        output_path = base_thumb_path

    try:
        clean_path = base_thumb_path.replace(".jpg", "_clean.jpg")
        source_path = clean_path if os.path.exists(clean_path) else base_thumb_path
        thumb = Image.open(source_path).convert("RGBA")
        logo = Image.open(logo_path)
        
        target_size = int(cfg.get("logo_size", 125))
        badge = create_circular_logo_badge(logo, target_size=target_size)
        
        pos = cfg.get("logo_position", "bottom_right")
        pad_x, pad_y = 35, 35
        
        if pos == "top_right":
            x = thumb.width - badge.width - pad_x
            y = pad_y
        elif pos == "bottom_left":
            x = pad_x
            y = thumb.height - badge.height - pad_y
        elif pos == "top_left":
            x = pad_x
            y = pad_y
        else: # default bottom_right (safe from top title text)
            x = thumb.width - badge.width - pad_x
            y = thumb.height - badge.height - pad_y

        thumb.paste(badge, (x, y), mask=badge)
        
        # Save back to RGB JPEG
        final_rgb = thumb.convert("RGB")
        final_rgb.save(output_path, "JPEG", quality=95)
        return output_path
    except Exception as e:
        print(f"[Branding] Failed to composite logo on thumbnail: {e}")
        return base_thumb_path
