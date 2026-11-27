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


def expect_model(page, row, method):
    page.wait_for_function(
        """method => {
            const dialog = document.getElementById('image-dialog');
            return dialog.dataset.state === 'ready' && dialog.dataset.method === method;
        }""", arg=method)
    image = page.locator("#dialog-image")
    assert image.get_attribute("src") == row["images"][method]
    assert image.evaluate("(img) => [img.naturalWidth, img.naturalHeight]") == [row["width"], row["height"]]
    assert page.locator("#dialog-title").inner_text() == row["title"]
    assert page.locator(".dialog-methods button[aria-pressed='true']").inner_text() == method
    return image


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
    screenshot = Image.open(io.BytesIO(stage.locator("canvas").screenshot())).convert("RGB")
    bounds = screenshot.getbbox()
    assert bounds and 3 < bounds[0] < bounds[2] < screenshot.width - 3
    assert 3 < bounds[1] < bounds[3] < screenshot.height - 3
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
    expect_model(page, row, "RiDEx")
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
    expect_model(page, row, "RiDEx")
    rendered = []
    media_box = page.locator(".dialog-media").bounding_box()
    for method in ("Base", "DiffusionNFT", "RiDEx"):
        button = page.locator(".dialog-methods").get_by_role("button", name=method, exact=True)
        (button.tap if mobile else button.click)()
        rendered.append(expect_model(page, row, method).screenshot())
        assert page.locator(".dialog-media").bounding_box() == media_box
    for first, second in zip(rendered, rendered[1:] + rendered[:1]):
        assert_canvas_pixels_changed(first, second)
    page.keyboard.press("ArrowLeft")
    expect_model(page, row, "DiffusionNFT")
    dialog.screenshot(path=str(OUTPUT / f"viewer_{name}.png"))
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
                                "canvas_center_open": True, "offscreen_frames_stopped": True,
                                "distinct_viewer_images": 3})
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
    expect_model(page, page.evaluate('window.RIDEX_EXAMPLES[14]'), 'RiDEx')
    page.locator('.dialog-methods button[data-method="Base"]').click()
    expect_model(page, page.evaluate('window.RIDEX_EXAMPLES[14]'), 'Base')
    page.wait_for_function('document.querySelectorAll(".dialog-images img").length === 1')
    page.locator('#close-dialog').click()
    report["auto_rotation"] = {"motion": "no-preference", "changes_before_pause": True, "stable_after_pause": True}
    context.close()


def verify_delayed_image(browser, report):
    context = browser.new_context(reduced_motion="reduce")
    page = context.new_page()
    page.goto(f"http://127.0.0.1:{SERVER.server_port}/", wait_until="domcontentloaded")
    row = page.evaluate("window.RIDEX_EXAMPLES.at(-1)")
    page.locator("#gallery-sphere").scroll_into_view_if_needed()
    blocked = []
    base_path = "**/" + row["images"]["Base"]
    page.route(base_path, lambda route: blocked.append(route))
    page.locator("#sphere-select").select_option(row["id"])
    page.locator("#sphere-open").click()
    expect_model(page, row, "RiDEx")
    page.locator('.dialog-methods button[data-method="Base"]').click()
    dialog = page.locator("#image-dialog")
    assert dialog.get_attribute("data-state") == "loading"
    assert dialog.get_attribute("data-method") == "RiDEx"
    page.locator('.dialog-methods button[data-method="DiffusionNFT"]').click()
    expect_model(page, row, "DiffusionNFT")
    assert len(blocked) == 1
    with page.expect_response("**/" + row["images"]["Base"]):
        blocked.pop().continue_()
    expect_model(page, row, "DiffusionNFT")

    page.locator("#close-dialog").click()
    row = page.evaluate("window.RIDEX_EXAMPLES.at(-2)")
    page.locator("#sphere-select").select_option(row["id"])
    page.locator("#sphere-open").click()
    expect_model(page, row, "RiDEx")
    path = "**/" + row["images"]["Base"]
    page.route(path, lambda route: route.abort())
    page.locator('.dialog-methods button[data-method="Base"]').click()
    page.wait_for_function('document.querySelector("#image-dialog").dataset.state === "error"')
    assert dialog.get_attribute("data-method") == "RiDEx"
    assert page.locator(".dialog-status").is_visible()
    page.unroute(path)
    page.locator('.dialog-methods button[data-method="Base"]').click()
    expect_model(page, row, "Base")

    path = "**/" + row["images"]["DiffusionNFT"]
    page.route(path, lambda route: blocked.append(route))
    page.locator('.dialog-methods button[data-method="DiffusionNFT"]').click()
    assert dialog.get_attribute("data-state") == "loading"
    page.keyboard.press("Escape")
    page.wait_for_function('document.activeElement.id === "sphere-open"')
    with page.expect_response(path):
        blocked.pop().continue_()
    assert not dialog.evaluate("element => element.open")
    assert page.locator("#dialog-image").count() == 0
    context.close()
    report["viewer_network"] = {
        "late_result_ignored": True, "failed_request_retriable": True,
        "close_during_load_restores_focus": True,
    }


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
                verify_delayed_image(browser, report)
            finally:
                browser.close()
        (OUTPUT / "gallery_sphere_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Gallery drag, tap, clipping, model switching and network checks passed.", flush=True)
    finally:
        SERVER.shutdown()
        SERVER.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
