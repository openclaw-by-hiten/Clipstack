import os
import json
import time
import shutil
import threading
from pathlib import Path
from core.config import WORKSPACE_DIR

PROFILES_DIR = WORKSPACE_DIR / "browser_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
]

def get_browser_executable() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

def get_status_file(platform: str) -> Path:
    return PROFILES_DIR / f"{platform}_status.json"

def get_profile_dir(platform: str) -> Path:
    return PROFILES_DIR / platform

def get_platform_status(platform: str) -> dict:
    status_file = get_status_file(platform)
    profile_dir = get_profile_dir(platform)
    
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            pass

    return {
        "platform": platform,
        "connected": False,
        "connecting": False,
        "account_name": "",
        "message": "Not connected. Click to link account."
    }

def get_all_browser_statuses() -> dict:
    return {
        "youtube": get_platform_status("youtube"),
        "instagram": get_platform_status("instagram"),
        "tiktok": get_platform_status("tiktok")
    }

def disconnect_platform(platform: str) -> dict:
    status_file = get_status_file(platform)
    profile_dir = get_profile_dir(platform)
    
    if status_file.exists():
        try:
            status_file.unlink()
        except Exception:
            pass

    if profile_dir.exists():
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except Exception:
            pass

    # Reset with explicit disconnected status
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "platform": platform,
            "connected": False,
            "connecting": False,
            "account_name": "",
            "message": "Disconnected successfully."
        }, f, indent=2)

    return {
        "platform": platform,
        "connected": False,
        "connecting": False,
        "account_name": "",
        "message": "Disconnected successfully."
    }

ACTIVE_LOGIN_TASKS = {}

def check_logged_in(platform: str, page, context) -> tuple[bool, str]:
    """
    Carefully checks whether the user has finished logging in.
    Must NOT trigger on initial login/redirect URLs.
    """
    try:
        url = page.url
        cookies = context.cookies()
        cookie_names = {c["name"] for c in cookies}

        if platform == "youtube":
            # The login page has URL on accounts.google.com
            is_on_accounts = "accounts.google.com" in url
            is_on_youtube = ("studio.youtube.com" in url or "youtube.com" in url or "myaccount.google.com" in url)
            
            # Auth cookies set by Google after sign-in:
            has_auth = bool({"SAPISID", "APISID", "SSID", "LOGIN_INFO"}.intersection(cookie_names))
            
            # Only genuine login if left accounts.google.com and reached youtube with auth
            if not is_on_accounts and is_on_youtube and has_auth:
                try:
                    title = page.title()
                    account_name = title.split("-")[0].strip() if title else "YouTube Creator"
                    if not account_name or "YouTube" in account_name:
                        account_name = "YouTube Channel"
                except Exception:
                    account_name = "YouTube Channel"
                return True, account_name

        elif platform == "instagram":
            is_on_login = ("accounts/login" in url or "accounts/onetap" in url or "accounts/emailsignup" in url)
            has_session = ("sessionid" in cookie_names and "ds_user_id" in cookie_names)
            
            if not is_on_login and has_session:
                ds_user = next((c.get("value") for c in cookies if c.get("name") == "ds_user_id"), "")
                return True, f"Instagram User #{ds_user}" if ds_user else "Instagram Profile"

        elif platform == "tiktok":
            is_on_login = ("/login" in url)
            has_session = ("sessionid" in cookie_names or "sessionid_ss" in cookie_names)
            
            if not is_on_login and has_session:
                return True, "TikTok Creator"

    except Exception:
        pass

    return False, ""

def start_interactive_login(platform: str) -> dict:
    if ACTIVE_LOGIN_TASKS.get(platform, {}).get("running"):
        return {"status": "in_progress", "message": "Login window is already open on your screen."}

    executable = get_browser_executable()
    if not executable:
        return {"status": "error", "message": "No Google Chrome or Microsoft Edge found on system."}

    profile_dir = get_profile_dir(platform)
    profile_dir.mkdir(parents=True, exist_ok=True)

    status_file = get_status_file(platform)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "platform": platform,
            "connected": False,
            "connecting": True,
            "message": "Login window opened. Please enter credentials & complete 2FA on screen."
        }, f, indent=2)

    def _login_worker():
        from playwright.sync_api import sync_playwright
        ACTIVE_LOGIN_TASKS[platform] = {"running": True}

        try:
            with sync_playwright() as p:
                context = p.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    executable_path=executable,
                    headless=False,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--start-maximized",
                        "--no-sandbox"
                    ],
                    ignore_default_args=["--enable-automation"],
                    viewport=None
                )
                page = context.pages[0] if context.pages else context.new_page()

                # Navigate to appropriate login URL
                if platform == "youtube":
                    page.goto("https://accounts.google.com/ServiceLogin?service=youtube&continue=https%3A%2F%2Fstudio.youtube.com%2F", timeout=60000)
                elif platform == "instagram":
                    page.goto("https://www.instagram.com/accounts/login/", timeout=60000)
                elif platform == "tiktok":
                    page.goto("https://www.tiktok.com/login", timeout=60000)

                page.bring_to_front()

                # Monitor for successful login for up to 5 minutes (300 seconds)
                start_time = time.time()
                logged_in = False
                account_name = ""

                while time.time() - start_time < 300:
                    time.sleep(1.5)
                    try:
                        # Check if user closed all pages
                        if not context.pages or page.is_closed():
                            break

                        logged_in, account_name = check_logged_in(platform, page, context)
                        if logged_in:
                            break
                    except Exception:
                        # If a transient navigation error happens, don't abort!
                        continue

                if logged_in:
                    # Save verified status
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "platform": platform,
                            "connected": True,
                            "connecting": False,
                            "account_name": account_name,
                            "connected_at": time.time(),
                            "message": f"Successfully connected as {account_name}!"
                        }, f, indent=2)
                    time.sleep(3)
                else:
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "platform": platform,
                            "connected": False,
                            "connecting": False,
                            "account_name": "",
                            "message": "Login was cancelled or window closed."
                        }, f, indent=2)

                try:
                    context.close()
                except Exception:
                    pass

        except Exception as e:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({
                    "platform": platform,
                    "connected": False,
                    "connecting": False,
                    "account_name": "",
                    "message": f"Login window closed: {str(e)}"
                }, f, indent=2)
        finally:
            ACTIVE_LOGIN_TASKS[platform] = {"running": False}

    thread = threading.Thread(target=_login_worker, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": f"Opened authentic {platform.title()} login window. Log in with your password and 2FA on screen."
    }
