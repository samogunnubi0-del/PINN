"""
Wake a sleeping Streamlit Community Cloud app using a headless browser.

HTTP GET pings do not start the Python process — Streamlit needs JS + WebSocket.
Used by .github/workflows/keepalive.yml on a schedule (every 6 hours).

Set STREAMLIT_APP_URL, e.g. https://your-app.streamlit.app
"""
from __future__ import annotations

import os
import sys
import time


def _app_url() -> str:
    url = (os.environ.get("STREAMLIT_APP_URL") or "").strip().rstrip("/")
    if not url.startswith("http"):
        print("ERROR: Set STREAMLIT_APP_URL to your public Streamlit Cloud URL.", file=sys.stderr)
        sys.exit(2)
    return url


def wake_streamlit_app(url: str, *, timeout_ms: int = 300_000) -> None:
    from playwright.sync_api import sync_playwright

    wake_labels = (
        "Yes, get this app back up!",
        "Yes, get this app back up",
        "Get app back up",
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"Opening {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        for label in wake_labels:
            btn = page.get_by_role("button", name=label)
            if btn.count() > 0:
                print(f"Sleep page detected — clicking: {label!r}")
                btn.first.click(timeout=30_000)
                break

        # Wait until main app shell or sidebar appears (loaded app).
        deadline = time.time() + (timeout_ms / 1000.0)
        last_err = None
        while time.time() < deadline:
            try:
                if page.locator("[data-testid='stAppViewContainer']").count() > 0:
                    print("App shell visible — wake successful.")
                    browser.close()
                    return
                if page.locator("[data-testid='stSidebar']").count() > 0:
                    print("Sidebar visible — wake successful.")
                    browser.close()
                    return
            except Exception as exc:
                last_err = exc
            page.wait_for_timeout(2000)

        browser.close()
        msg = "Timed out waiting for Streamlit app to load after wake."
        if last_err:
            msg += f" Last error: {last_err}"
        raise TimeoutError(msg)


def main() -> None:
    url = _app_url()
    wake_streamlit_app(url)
    print("Keepalive finished OK.")


if __name__ == "__main__":
    main()
