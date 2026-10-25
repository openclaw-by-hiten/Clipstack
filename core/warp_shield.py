import os
import sys
import time
import socket
import shutil
import tarfile
import zipfile
import platform
import atexit
import threading
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional
from core.config import WORKSPACE_DIR

SHIELD_DIR = WORKSPACE_DIR / "warp_shield"
SHIELD_DIR.mkdir(parents=True, exist_ok=True)

WARP_SOCKS_PORT = 40000
WARP_HTTP_PORT = 40001
WARP_HTTP_URL = f"http://127.0.0.1:{WARP_HTTP_PORT}"

_SHIELD_LOCK = threading.Lock()
_WIREPROXY_PROC: Optional[subprocess.Popen] = None
_TUNNEL_VERIFIED: bool = False

WGCF_VERSION = "2.3.1"
WIREPROXY_VERSION = "1.1.3"
WARP_ENDPOINTS = [
    None,                      # Default from wgcf (engage.cloudflareclient.com:2408)
    "162.159.192.1:2408",      # Direct IPv4 UDP 2408 (avoids DNS issues)
    "162.159.192.1:4500",      # Fallback NAT-T UDP 4500
    "162.159.192.1:500",       # Fallback IKE UDP 500
]


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _verify_proxy_tunnel(proxy_url: str, timeout: float = 7.0) -> bool:
    """Confirms end-to-end WireGuard handshake with Cloudflare edge (`warp=on`)."""
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
        req = urllib.request.Request(
            "https://cloudflare.com/cdn-cgi/trace",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return "warp=on" in body or "warp=plus" in body
    except Exception:
        return False


def _detect_platform_arch() -> tuple[str, str]:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        arch = "amd64"

    if "windows" in sys_name or os.name == "nt":
        os_key = "windows"
    elif "darwin" in sys_name:
        os_key = "darwin"
    else:
        os_key = "linux"
    return os_key, arch


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=35) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _ensure_binaries() -> tuple[Path, Path]:
    os_key, arch = _detect_platform_arch()
    exe_suffix = ".exe" if os_key == "windows" else ""

    wgcf_bin = SHIELD_DIR / f"wgcf{exe_suffix}"
    wireproxy_bin = SHIELD_DIR / f"wireproxy{exe_suffix}"

    if not wgcf_bin.exists() or wgcf_bin.stat().st_size < 100_000:
        wgcf_asset = f"wgcf_{WGCF_VERSION}_{os_key}_{arch}{exe_suffix}"
        wgcf_url = f"https://github.com/ViRb3/wgcf/releases/download/v{WGCF_VERSION}/{wgcf_asset}"
        print(f"[WarpShield] Downloading {wgcf_asset}...")
        _download_file(wgcf_url, wgcf_bin)
        if os_key != "windows":
            os.chmod(wgcf_bin, 0o755)

    if not wireproxy_bin.exists() or wireproxy_bin.stat().st_size < 100_000:
        wp_asset = f"wireproxy_{os_key}_{arch}.tar.gz"
        wp_url = f"https://github.com/windtf/wireproxy/releases/download/v{WIREPROXY_VERSION}/{wp_asset}"
        tgz_path = SHIELD_DIR / wp_asset
        print(f"[WarpShield] Downloading {wp_asset}...")
        _download_file(wp_url, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(str(SHIELD_DIR))
        try:
            tgz_path.unlink()
        except OSError:
            pass
        if not wireproxy_bin.exists():
            for cand in SHIELD_DIR.rglob(f"wireproxy{exe_suffix}"):
                if cand.is_file() and cand != wireproxy_bin:
                    shutil.move(str(cand), str(wireproxy_bin))
                    break
        if os_key != "windows" and wireproxy_bin.exists():
            os.chmod(wireproxy_bin, 0o755)

    return wgcf_bin, wireproxy_bin


def _ensure_warp_config(
    wgcf_bin: Path,
    force_regenerate: bool = False,
    endpoint_override: Optional[str] = None,
) -> Path:
    account_toml = SHIELD_DIR / "wgcf-account.toml"
    profile_conf = SHIELD_DIR / "wgcf-profile.conf"
    wp_conf = SHIELD_DIR / "wireproxy.conf"

    if force_regenerate:
        for f in (account_toml, profile_conf, wp_conf):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

    if not account_toml.exists():
        print("[WarpShield] Provisioning free Cloudflare WARP identity...")
        last_err = None
        for attempt in range(3):
            try:
                subprocess.run(
                    [str(wgcf_bin), "register", "--accept-tos"],
                    cwd=str(SHIELD_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=25,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        if last_err is not None and not account_toml.exists():
            raise RuntimeError(f"wgcf register failed: {last_err}")

    if not profile_conf.exists():
        subprocess.run(
            [str(wgcf_bin), "generate"],
            cwd=str(SHIELD_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=15,
        )

    wg_raw = profile_conf.read_text(encoding="utf-8")
    cleaned_lines = []
    for line in wg_raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("DNS ="):
            cleaned_lines.append("DNS = 1.1.1.1")
        elif endpoint_override and stripped.startswith("Endpoint ="):
            cleaned_lines.append(f"Endpoint = {endpoint_override}")
        else:
            cleaned_lines.append(line)

    wp_text = (
        "\n".join(cleaned_lines)
        + f"\n\n[Socks5]\nBindAddress = 127.0.0.1:{WARP_SOCKS_PORT}\n"
        + f"\n[HTTP]\nBindAddress = 127.0.0.1:{WARP_HTTP_PORT}\n"
    )
    wp_conf.write_text(wp_text, encoding="utf-8")
    return wp_conf


def stop_warp_proxy() -> None:
    global _WIREPROXY_PROC, _TUNNEL_VERIFIED
    _TUNNEL_VERIFIED = False
    if _WIREPROXY_PROC is not None:
        try:
            _WIREPROXY_PROC.terminate()
            _WIREPROXY_PROC.wait(timeout=3)
        except Exception:
            try:
                _WIREPROXY_PROC.kill()
            except Exception:
                pass
        _WIREPROXY_PROC = None


atexit.register(stop_warp_proxy)


def _kill_orphan_wireproxy() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "wireproxy.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            ["pkill", "-f", "wireproxy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(0.4)


def ensure_warp_proxy() -> Optional[str]:
    """
    Ensures a userspace Cloudflare WARP proxy (wireproxy) is listening on 127.0.0.1:40001
    and passing traffic (`warp=on`). Requires zero root, zero kernel modules, and zero cookies.
    """
    global _WIREPROXY_PROC, _TUNNEL_VERIFIED
    if _is_port_open(WARP_HTTP_PORT):
        if _TUNNEL_VERIFIED or _verify_proxy_tunnel(WARP_HTTP_URL):
            _TUNNEL_VERIFIED = True
            return WARP_HTTP_URL

    with _SHIELD_LOCK:
        if _is_port_open(WARP_HTTP_PORT):
            if _TUNNEL_VERIFIED or _verify_proxy_tunnel(WARP_HTTP_URL):
                _TUNNEL_VERIFIED = True
                return WARP_HTTP_URL
            _kill_orphan_wireproxy()

        try:
            wgcf_bin, wireproxy_bin = _ensure_binaries()

            for ep in WARP_ENDPOINTS:
                wp_conf = _ensure_warp_config(
                    wgcf_bin,
                    force_regenerate=False,
                    endpoint_override=ep,
                )
                stop_warp_proxy()
                _WIREPROXY_PROC = subprocess.Popen(
                    [str(wireproxy_bin), "-c", str(wp_conf.name)],
                    cwd=str(SHIELD_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                port_ready = False
                for _ in range(18):
                    if _is_port_open(WARP_HTTP_PORT):
                        port_ready = True
                        break
                    time.sleep(0.35)

                if port_ready and _verify_proxy_tunnel(WARP_HTTP_URL):
                    _TUNNEL_VERIFIED = True
                    ep_label = ep or "engage.cloudflareclient.com:2408"
                    print(f"[WarpShield] Cloudflare WARP tunnel verified on {WARP_HTTP_URL} ({ep_label})")
                    return WARP_HTTP_URL

                stop_warp_proxy()
        except Exception as e:
            print(f"[WarpShield] Warning: Could not start WARP shield ({e}). Falling back to direct connection.")

    return None


def rotate_warp_proxy() -> Optional[str]:
    """Rotates the Cloudflare WARP identity to obtain a fresh clean exit IP if rate-limited."""
    global _WIREPROXY_PROC, _TUNNEL_VERIFIED
    with _SHIELD_LOCK:
        try:
            print("[WarpShield] Rotating Cloudflare WARP identity for fresh IP...")
            stop_warp_proxy()
            _kill_orphan_wireproxy()

            wgcf_bin, wireproxy_bin = _ensure_binaries()
            for idx, ep in enumerate(WARP_ENDPOINTS):
                wp_conf = _ensure_warp_config(
                    wgcf_bin,
                    force_regenerate=(idx == 0),
                    endpoint_override=ep,
                )
                _WIREPROXY_PROC = subprocess.Popen(
                    [str(wireproxy_bin), "-c", str(wp_conf.name)],
                    cwd=str(SHIELD_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                port_ready = False
                for _ in range(18):
                    if _is_port_open(WARP_HTTP_PORT):
                        port_ready = True
                        break
                    time.sleep(0.35)

                if port_ready and _verify_proxy_tunnel(WARP_HTTP_URL):
                    _TUNNEL_VERIFIED = True
                    print(f"[WarpShield] Rotated WARP tunnel verified on {WARP_HTTP_URL}")
                    return WARP_HTTP_URL

                stop_warp_proxy()
        except Exception as e:
            print(f"[WarpShield] Rotation warning: {e}")
    return None


def get_js_runtime_opts() -> dict:
    """
    Ensures a JavaScript runtime (Node.js or Deno) is available for yt-dlp EJS signature solving.
    Auto-downloads standalone Deno on Linux/Colab if neither Node nor Deno is installed.
    """
    if os.name == "nt" and os.path.exists(r"C:\Program Files\nodejs\node.exe"):
        return {
            "js_runtimes": {"node": {"path": r"C:\Program Files\nodejs\node.exe"}},
            "remote_components": ["ejs:github"],
        }

    node_bin = shutil.which("node")
    if node_bin and os.path.exists(node_bin):
        return {
            "js_runtimes": {"node": {"path": node_bin}},
            "remote_components": ["ejs:github"],
        }

    deno_bin = shutil.which("deno")
    if deno_bin and os.path.exists(deno_bin):
        return {
            "js_runtimes": {"deno": {"path": deno_bin}},
            "remote_components": ["ejs:github"],
        }

    # Auto-provision standalone Deno on Linux x86_64 (e.g. Google Colab) if no JS runtime exists
    os_key, arch = _detect_platform_arch()
    local_deno = SHIELD_DIR / ("deno.exe" if os_key == "windows" else "deno")
    if not local_deno.exists() and os_key == "linux" and arch == "amd64":
        try:
            print("[WarpShield] Provisioning standalone Deno JS runtime for YouTube challenge solving...")
            zip_path = SHIELD_DIR / "deno.zip"
            _download_file(
                "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip",
                zip_path,
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(SHIELD_DIR))
            zip_path.unlink(missing_ok=True)
            os.chmod(local_deno, 0o755)
        except Exception as e:
            print(f"[WarpShield] Deno auto-install warning: {e}")

    if local_deno.exists():
        return {
            "js_runtimes": {"deno": {"path": str(local_deno)}},
            "remote_components": ["ejs:github"],
        }

    return {}


def get_proxy_env(proxy_url: Optional[str]) -> dict:
    env = dict(os.environ)
    if proxy_url:
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
    return env
