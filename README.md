# RiDEx project page

[Live website](https://shawn-codedev.github.io/RiDEX/)

A buildless research page: author-first hero, the supplied four-task training
comparison PDF, and matched Base / DiffusionNFT / RiDEx image comparisons.
No vector method explainer, duplicated result tables, or frontend framework.

## Assets

- `assets/training_comparison.pdf` is the supplied paper figure, unchanged.
  The WebP preview is rendered directly from the PDF without redrawing data.
- Individual native-resolution images replace compressed paper-plate crops.
  Every triplet uses the same prompt, seed, resolution, 40-step Euler sampler
  and CFG 4.5. Click an image to inspect it and switch methods.
- `assets/comparison_manifest.json` records settings and checkpoint identities.
  Examples are selected, not aggregate win-rate evidence.
- Native-1024 preference examples are new clean checkpoint inference. Alignment
  examples retain native-512 original outputs. No AI upscaling or retouching.
- The hero is a decorative crop of a real RiDEx output. Comparisons show full frames.

## Publishing

GitHub Pages serves `main` from the repository root, without a build step or
external JavaScript/font dependencies. `.nojekyll` is retained. Serve with any
static HTTP server, or open `index.html` directly.

On an HPC cluster, image processing, inference and browser tests belong on
allocated compute nodes. This repository contains no weights or credentials.
