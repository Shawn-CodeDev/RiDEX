"""Focused desktop and touch regression checks for the Three.js sphere gallery."""
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import io
import json
import os
from pathlib import Path
import socket
import threading

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("RIDEX_SITE_ROOT", HERE.parents[2])).resolve()
OUTPUT = Path(os.environ.get("RIDEX_QA_OUTPUT", HERE.parents[3] / "qa" / "globe_20261127")).resolve()


def assert_canvas_pixels_changed(before, after):
    first = Image.open(io.BytesIO(before)).convert("RGB")
    second = Image.open(io.BytesIO(after)).convert("RGB")
    extrema = first.getextrema()
    assert any(low != high for low, high in extrema), "Sphere canvas screenshot is blank"
    assert ImageChops.difference(first, second).getbbox(), "Sphere canvas pixels did not change"


def verify_viewport(browser, name, width, height, report, mobile=False):
    context = browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1,
        is_mobile=mobile,
        has_touch=mobile,
        reduced_motion="reduce",
    )
    page = context.new_page()
    page_errors, console_errors, failed_responses = [], [], []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("requestfailed", lambda request: failed_responses.append(f"{request.url}: {request.failure}"))
    page.on("response", lambda response: failed_responses.append(f"{response.status} {response.url}") if response.status >= 400 else None)
    page.goto(f"http://127.0.0.1:{SERVER.server_port}/", wait_until="networkidle")

    stage = page.locator("#gallery-sphere")
    stage.scroll_into_view_if_needed()
    page.wait_for_function("document.querySelector('#gallery-sphere')?.dataset.state === 'ready'", timeout=60_000)
    page.wait_for_function("window.RIDEX_EXAMPLES?.length > 0")
    expected_count = page.evaluate("window.RIDEX_EXAMPLES.length")
    assert stage.locator("canvas").count() == 1
    assert int(stage.get_attribute("data-loaded-count")) == expected_count
    stage.locator("..").screenshot(path=str(OUTPUT / f"sphere_initial_{name}.png"))

    select = page.locator("#sphere-select")
    assert select.evaluate("el => el instanceof HTMLSelectElement")
    assert select.locator("option").count() == expected_count
    selected_id = select.locator("option").nth(0).get_attribute("value")
    assert selected_id == page.evaluate("window.RIDEX_EXAMPLES[0].id")
    assert stage.get_attribute("data-selected-id") == selected_id
    assert page.locator("#sphere-title").inner_text().strip()

    toggle = page.locator("#sphere-toggle")
    assert toggle.get_attribute("aria-pressed") == "false", "Reduced-motion preference should start with auto rotation paused"
    toggle.click()
    assert toggle.get_attribute("aria-pressed") == "true"
    toggle.click()
    assert toggle.get_attribute("aria-pressed") == "false"

    canvas = stage.locator("canvas")
    box = canvas.bounding_box()
    assert box and box["width"] > 10 and box["height"] > 10
    if mobile:
        client = context.new_cdp_session(page)
        sx, sy = box["x"] + box["width"] * 0.72, box["y"] + box["height"] * 0.52
        ex, ey = sx - min(110, box["width"] * 0.3), sy + 12
        before = canvas.screenshot()
        client.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": sx, "y": sy, "id": 1}]})
        client.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": ex, "y": ey, "id": 1}]})
        client.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        page.wait_for_timeout(150)
        assert_canvas_pixels_changed(before, canvas.screenshot())
        assert page.locator("#image-dialog").evaluate("dialog => dialog.open") is False
    else:
        before = canvas.screenshot()
        page.mouse.move(box["x"] + box["width"] * 0.72, box["y"] + box["height"] * 0.52)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.42, box["y"] + box["height"] * 0.58, steps=8)
        page.mouse.up()
        page.wait_for_timeout(100)
        assert_canvas_pixels_changed(before, canvas.screenshot())
        assert page.locator("#image-dialog").evaluate("dialog => dialog.open") is False

    if expected_count > 1:
        selected_id = page.evaluate("window.RIDEX_EXAMPLES.at(-1).id")
        select.select_option(selected_id)
        assert stage.get_attribute("data-selected-id") == selected_id
    page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    row = page.evaluate("id => window.RIDEX_EXAMPLES.find(example => example.id === id)", selected_id)
    assert page.locator("#sphere-title").inner_text().strip() == row["title"]

    # A click/tap at the selected card center should open the matching example.
    stage.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    center_x, center_y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    if mobile:
        client.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": center_x, "y": center_y, "id": 1}]})
        client.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    else:
        page.mouse.click(center_x, center_y)
    dialog = page.locator("#image-dialog")
    assert dialog.evaluate("element => element.open"), "Canvas center hit did not open the selected example"
    image = page.locator("#dialog-image")
    page.wait_for_function("img => img.complete && img.naturalWidth > 0", arg=image.element_handle())
    assert image.get_attribute("src").endswith(row["images"]["RiDEx"])
    page.locator("#close-dialog").click()

    # Reset and keyboard input should remain usable while reduced motion is enabled.
    page.locator("#sphere-reset").click()
    page.locator("canvas").focus()
    before_key = canvas.screenshot()
    page.keyboard.press("ArrowRight")
    page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert_canvas_pixels_changed(before_key, canvas.screenshot())

    page.locator("#sphere-open").click()
    assert dialog.evaluate("element => element.open")
    page.wait_for_function("img => img.complete && img.naturalWidth > 0", arg=image.element_handle())
    assert image.get_attribute("src").endswith(row["images"]["RiDEx"])
    for method in ("Base", "DiffusionNFT", "RiDEx"):
        page.locator(".dialog-methods button").filter(has_text=method).click()
        assert page.locator("#dialog-title").inner_text().startswith(method)
        assert page.locator(".dialog-methods button[aria-pressed='true']").inner_text() == method
        assert image.get_attribute("src").endswith(row["images"][method])
        image.evaluate("image => image.decode()")
    page.locator("#close-dialog").click()
    assert not dialog.evaluate("element => element.open")

    stage.locator("..").screenshot(path=str(OUTPUT / f"sphere_{name}.png"))
    page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
    page.wait_for_timeout(150)
    frames_stopped = page.evaluate("Number(document.querySelector('#gallery-sphere').dataset.frames)")
    page.wait_for_timeout(500)
    assert page.evaluate("Number(document.querySelector('#gallery-sphere').dataset.frames)") == frames_stopped, "Sphere kept rendering while offscreen"
    overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
    assert not overflow, f"Horizontal overflow at {width}px"
    if page_errors or console_errors or failed_responses:
        raise AssertionError({"page_errors": page_errors, "console_errors": console_errors,
                              "failed_assets": failed_responses})
    report["viewports"].append({"name": name, "width": width, "height": height, "overflow": overflow,
                                "examples": expected_count, "touch": mobile, "canvas_drag": True,
                                "canvas_center_open": True, "offscreen_frames_stopped": True})
    context.close()


def verify_auto_rotation(browser, report):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="no-preference")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.goto(f"http://127.0.0.1:{SERVER.server_port}/", wait_until="networkidle")
    stage = page.locator("#gallery-sphere")
    stage.scroll_into_view_if_needed()
    page.wait_for_function("document.querySelector('#gallery-sphere')?.dataset.state === 'ready'", timeout=60_000)
    canvas = stage.locator("canvas")
    before = canvas.screenshot()
    page.wait_for_timeout(700)
    moving = canvas.screenshot()
    assert_canvas_pixels_changed(before, moving)
    toggle = page.locator("#sphere-toggle")
    assert toggle.get_attribute("aria-pressed") == "true"
    toggle.click()
    assert toggle.get_attribute("aria-pressed") == "false"
    stopped = canvas.screenshot()
    page.wait_for_timeout(500)
    assert canvas.screenshot() == stopped, "Sphere continued rotating after Pause"
    assert not errors, errors
    select = page.locator('#sphere-select')
    select.select_option(page.evaluate('window.RIDEX_EXAMPLES[14].id'))
    page.wait_for_timeout(850)
    stage.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
    assert page.locator('#image-dialog').evaluate('dialog => dialog.open')
    assert page.locator('#dialog-image').get_attribute('src') == page.evaluate('window.RIDEX_EXAMPLES[14].images.RiDEx')
    page.locator('#close-dialog').click()
    report["auto_rotation"] = {"motion": "no-preference", "changes_before_pause": True, "stable_after_pause": True}
    context.close()


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().lower().startswith("login"):
        raise RuntimeError("Run gallery_sphere.py only on a non-login host inside a SLURM allocation")
    if not (ROOT / "index.html").is_file():
        raise FileNotFoundError(f"RIDEX_SITE_ROOT must contain index.html: {ROOT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    global SERVER
    SERVER = HTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT)))
    thread = threading.Thread(target=SERVER.serve_forever, daemon=True)
    thread.start()
    report = {"root": str(ROOT), "viewports": []}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=[
                "--no-sandbox", "--disable-dev-shm-usage", "--renderer-process-limit=1",
                "--num-raster-threads=1", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                "--disable-features=Vulkan",
            ])
            try:
                verify_viewport(browser, "desktop1440x1000", 1440, 1000, report)
                verify_viewport(browser, "touchmobile390x844", 390, 844, report, mobile=True)
                verify_viewport(browser, "small320x800", 320, 800, report)
                verify_auto_rotation(browser, report)
            finally:
                browser.close()
        (OUTPUT / "gallery_sphere_report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        SERVER.shutdown()
        SERVER.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
