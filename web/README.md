# Project page

GitHub Pages serves the repository root at <https://gary.ac.cn/RiDEX/>. The entry point is [index.html](../index.html); there is no build step.

| Location | Contents |
| --- | --- |
| `web/` | Styles, navigation, gallery interactions and site icon |
| [assets/examples/](../assets/examples/) | Native-aspect comparison images, grouped by filename into matched triplets |
| [assets/examples.js](../assets/examples.js) | Gallery titles, prompts, seeds, sizes and image paths for the browser |
| [assets/comparison_manifest.json](../assets/comparison_manifest.json) | Generation settings, checkpoint identities and selection criteria |
| [assets/training_comparison.pdf](../assets/training_comparison.pdf) | Paper figure; the adjacent WebP files are its desktop and mobile previews |
| [assets/fonts/](../assets/fonts/) | Self-hosted fonts and their OFL notices |
| [assets/globe/](../assets/globe/) | Native-aspect 384-pixel gallery textures; full-resolution images remain unchanged |
| [web/gallery-sphere.js](gallery-sphere.js) | Three.js sphere, mouse/touch rotation, keyboard navigation and image selection |
| [web/vendor/](vendor/) | Self-hosted Three.js 0.180.0 and OrbitControls; license in `licenses/threejs.txt` |

Serve the repository root over HTTP to preview the ES-module gallery:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000/>. Keep site links relative so the page works under the hosted `/RiDEX/` path. Preserve published image and figure URLs when changing the layout. The hero is [editorial artwork](../assets/hero_art_manifest.json), separate from the experiment gallery.

The sphere loads when approaching the gallery, with four concurrent thumbnail
requests. Rotation pauses while dragging or viewing a full-resolution image;
off-screen and hidden-tab rendering stops. Reduced-motion preferences disable
automatic rotation. The native example selector and image viewer remain usable
when WebGL is unavailable. No external CDN is needed.

After adding examples, run `python web/prepare_globe.py` inside a compute-node
allocation. Run the focused desktop/touch regression with
`python web/tests/gallery_sphere.py` in an allocated environment with Playwright,
Chromium and Pillow installed. `RIDEX_QA_OUTPUT` sets the screenshot/report folder.
