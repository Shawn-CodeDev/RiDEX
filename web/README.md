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

To preview, open `index.html` in a browser. Alternatively, from the repository root on a development machine:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000/>. Keep site links relative so the page works under the hosted `/RiDEX/` path. Preserve published image and figure URLs when changing the layout. The hero is [editorial artwork](../assets/hero_art_manifest.json), separate from the experiment gallery.
