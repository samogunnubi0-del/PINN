"""
Wake a sleeping Streamlit Community Cloud app using a headless browser.

HTTP GET pings do not start the Python process — Streamlit needs JS + WebSocket.
Used by .github/workflows/keepalive.yml on a schedule (every 6 hours).

Set STREAMLIT_APP_URL, e.g. https://your-app.streamlit.app

Success = browser visit + optional wake click + hold WebSocket open long enough
for the container to start. We do NOT require full PyTorch/model render (can exceed
10 min on free tier); that was causing false CI failures.
"""
from __future__ import annotations

import os
import sys
import time

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

WAKE_BUTTON_NAMES = (
    "Yes, get this app back up!",
    "Yes, get this app back up",
    "Get app back up",
)

APP_MARKERS = (
    "IsotopePINN",
    "6/6 PASS",
    "Ac-225 Production Surrogate",
)

SLEEP_MARKERS = (
    "Zzzz",
    "This app has gone to sleep",
    "inactive for too long",
    "get this app back up",
)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _app_url() -> str:
    url = (os.environ.get("STREAMLIT_APP_URL") or "").strip().rstrip("/")
    if not url.startswith("http"):
        print("ERROR: Set STREAMLIT_APP_URL to your public Streamlit Cloud URL.", file=sys.stderr)
        sys.exit(2)
    return url


def _page_text(page) -> str:
    try:
        return (page.inner_text("body") or "").lower()
    except Exception:
        return ""


def _find_wake_button(page):
    """Return a clickable wake button on page or any child frame."""
    for label in WAKE_BUTTON_NAMES:
        btn = page.get_by_role("button", name=label)
        if btn.count() > 0:
            return btn.first, label

    # Partial match fallback (Streamlit copy changes occasionally).
    for frame in page.frames:
        try:
            buttons = frame.locator("button")
            n = buttons.count()
            for i in range(min(n, 20)):
                text = (buttons.nth(i).inner_text() or "").strip()
                lower = text.lower()
                if "back up" in lower or "wake" in lower:
                    return buttons.nth(i), text
        except Exception:
            continue
    return None, ""


def _app_loaded(page) -> bool:
    """Best-effort signal that the Streamlit shell or app content is visible."""
    selectors = (
        "[data-testid='stAppViewContainer']",
        "[data-testid='stSidebar']",
        "[data-testid='stMainBlockContainer']",
    )
    for frame in page.frames:
        for sel in selectors:
            try:
                if frame.locator(sel).count() > 0:
                    return True
            except Exception:
                continue

    text = _page_text(page)
    return any(marker.lower() in text for marker in APP_MARKERS)


def _looks_asleep(page) -> bool:
    text = _page_text(page)
    return any(marker.lower() in text for marker in SLEEP_MARKERS)


def _log_page_state(page, *, prefix: str = "") -> None:
    try:
        print(f"{prefix}URL: {page.url}")
        print(f"{prefix}Title: {page.title()}")
    except Exception as exc:
        print(f"{prefix}Could not read page metadata: {exc}")

    text = _page_text(page)
    snippet = " ".join(text.split())[:400]
    print(f"{prefix}Body snippet: {snippet!r}")
    print(f"{prefix}Frames: {len(page.frames)}")


def wake_streamlit_app(
    url: str,
    *,
    goto_timeout_ms: int = 120_000,
    post_wake_wait_s: int = 120,
    poll_timeout_s: int = 180,
) -> None:
    from playwright.sync_api import sync_playwright

    woke = False
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        print(f"Opening {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout_ms)

        # Let Streamlit sleep/wake UI render before probing.
        page.wait_for_timeout(5_000)

        btn, label = _find_wake_button(page)
        if btn is not None:
            print(f"Sleep page detected — clicking: {label!r}")
            btn.click(timeout=30_000)
            woke = True
            page.wait_for_timeout(3_000)
        elif _app_loaded(page):
            print("App already loaded — no wake click needed.")
        elif _looks_asleep(page):
            print("Sleep markers visible but wake button not found; holding connection anyway.")
        else:
            print("No wake button; treating visit as keepalive ping (container may be starting).")

        deadline = time.time() + poll_timeout_s
        while time.time() < deadline:
            if _app_loaded(page):
                print("App content detected — keepalive successful.")
                browser.close()
                return
            page.wait_for_timeout(3_000)

        # Hold WebSocket after wake so the Python process can start (PyTorch cold start).
        hold_s = post_wake_wait_s if woke else min(post_wake_wait_s, 60)
        print(f"Holding browser session {hold_s}s so Streamlit can finish starting...")
        page.wait_for_timeout(hold_s * 1000)

        if _app_loaded(page):
            print("App content detected after hold — keepalive successful.")
        else:
            print(
                "Full UI not confirmed (PyTorch cold start may still be running). "
                "Visit + wake click counts as success for scheduled keepalive."
            )

        _log_page_state(page, prefix="Final state — ")
        browser.close()


def main() -> None:
    url = _app_url()
    goto_timeout_ms = _env_int("KEEPALIVE_GOTO_TIMEOUT_MS", 120_000)
    post_wake_wait_s = _env_int("KEEPALIVE_POST_WAKE_WAIT_S", 120)
    poll_timeout_s = _env_int("KEEPALIVE_POLL_TIMEOUT_S", 180)

    wake_streamlit_app(
        url,
        goto_timeout_ms=goto_timeout_ms,
        post_wake_wait_s=post_wake_wait_s,
        poll_timeout_s=poll_timeout_s,
    )
    print("Keepalive finished OK.")


if __name__ == "__main__":
    main()
