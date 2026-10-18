"""Optional end-to-end check against a running MOCK_MODEL=1 server.

Run: python tests/browser_smoke.py [--base-url http://127.0.0.1:8000]
Requires Playwright and a previously installed Chromium browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / ".verification"
    output.mkdir(exist_ok=True)
    video = output / "ui-sample.mov"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=640x360:rate=12", "-t", "9",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-f", "mov", str(video),
    ], check=True, timeout=30)
    errors, failures, external, requests, canceled_media = [], [], [], [], []
    origin = urlsplit(args.base_url).netloc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, locale="ru-RU")
        page = context.new_page()
        page.on("pageerror", lambda failure: errors.append(str(failure)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        def record_failure(request):
            failure = {"url": request.url, "error": request.failure}
            # Reloading the page cancels the local video blob. This is expected
            # browser lifecycle behavior; HTTP/asset failures still fail QA.
            if request.url.startswith("blob:") and request.failure == "net::ERR_ABORTED":
                canceled_media.append(failure)
            else:
                failures.append(failure)

        page.on("requestfailed", record_failure)

        def record(request):
            requests.append(request.url)
            parsed = urlsplit(request.url)
            if parsed.scheme in {"http", "https"} and parsed.netloc != origin:
                external.append(request.url)
            assert "x-api-key" not in request.headers, "The browser must never receive/use the API key"

        page.on("request", record)
        page.goto(args.base_url, wait_until="networkidle")
        expect(page.get_by_role("heading", name="Проверка видеозаписи кассира")).to_be_visible()
        expect(page.locator("#mock-notice")).to_be_visible()
        page.screenshot(path=str(output / "ui-empty.png"), full_page=True)
        page.locator("#file-input").set_input_files(video)
        expect(page.locator("#video-player")).to_be_visible()
        expect(page.locator("#file-name")).to_have_text(video.name)
        page.get_by_role("button", name="Проверить видео").click()
        expect(page.locator("#processing-state")).to_be_visible()
        expect(page.locator("#analyze-button")).to_be_disabled()
        expect(page.locator("#verdict-text")).to_have_text("Обнаружены признаки возможных нарушений", timeout=25000)
        expect(page.locator(".violation-card")).to_have_count(2)
        expect(page.locator(".violation-card > .violation-head + .score-line + .violation-description")).to_have_count(2)
        expect(page.locator(".frame-button")).to_have_count(6)
        expect(page.locator(".rule-block")).to_have_count(2)
        expect(page.locator(".rule-block dd.is-empty")).to_have_text("источник не указан")
        expect(page.locator(".timeline-marker")).to_have_count(2)
        expect(page.locator("#analyze-button")).to_be_enabled()
        expect(page.locator(".playback-note")).to_be_hidden()
        page.locator(".timecode").first.click()
        seeked = page.locator("#video-player").evaluate("video => video.currentTime")
        assert seeked > 0
        page.locator(".frame-button").first.click()
        expect(page.locator("#frame-dialog")).to_be_visible()
        expect(page.locator("#enlarged-frame")).to_be_visible()
        assert page.locator("#enlarged-frame").evaluate("image => image.naturalWidth") > 0
        page.screenshot(path=str(output / "ui-evidence.png"), full_page=True)
        page.get_by_role("button", name="Закрыть увеличенный кадр").click()
        expect(page.locator("#frame-dialog")).to_be_hidden()
        page.screenshot(path=str(output / "ui-result.png"), full_page=True)
        page.set_viewport_size({"width": 360, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Horizontal overflow on mobile"
        page.screenshot(path=str(output / "ui-mobile.png"), full_page=True)
        page.locator(".violation-card").first.screenshot(path=str(output / "ui-card-360.png"))

        # Reset to verify keyboard picker and browser-side validation as well.
        page.reload(wait_until="networkidle")
        page.locator("#file-input").set_input_files({"name": "wrong.txt", "mimeType": "text/plain", "buffer": b"invalid"})
        expect(page.locator("#upload-error")).to_be_visible()
        assert not external, external
        assert not errors, errors
        assert not failures, failures
        summary = {"passed": True, "mock": True, "violations": 2, "evidence_frames": 6,
                   "mobile_width": 360, "horizontal_overflow": False, "score_before_description": True,
                   "seek_seconds": seeked, "external_requests": external,
                   "browser_errors": errors, "failed_requests": failures,
                   "canceled_local_media": len(canceled_media),
                   "requests": requests, "screenshots": [str(path.name) for path in output.glob("ui-*.png")]}
        (output / "browser-report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "requests"}, ensure_ascii=False))
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
