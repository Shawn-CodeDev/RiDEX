# RiDEx project page

[Live website](https://gary.ac.cn/RiDEX/)

A buildless research page: an interactive full-screen hero, the supplied four-task training
comparison PDF, and matched Base / DiffusionNFT / RiDEx image comparisons.
No vector method explainer, duplicated result tables, or frontend framework.

## Interactive hero

The full-screen image pairs native-1024 Base and RiDEx generations with the
same prompt and seed. A soft cursor-following CSS mask reveals RiDEx. Base,
Spotlight and RiDEx controls also work with touch and keyboard. The author
list and affiliations remain visible in the first section.

`hero.css` and `hero.js` own this interaction separately from the image gallery.
Animation frames run only until the cursor settles, and stop offscreen or when
the page is hidden. Reduced-motion settings disable intro animation and easing.
Inter and italic Playfair Display are self-hosted with their OFL licenses.

## Assets

- `assets/training_comparison.pdf` is the supplied paper figure, unchanged.
  The WebP preview is rendered directly from the PDF without redrawing data.
- Individual native-resolution images replace compressed paper-plate crops.
  Every triplet uses the same prompt, seed, resolution, 40-step Euler sampler
  and CFG 4.5. Click an image to inspect it and switch methods.
- `assets/comparison_manifest.json` records settings and checkpoint identities.
  Examples are selected, not aggregate win-rate evidence.
- The gallery contains 7 visually curated matched triplets (21 images), with
  square, landscape and portrait filters and a clickable image wall. After review
  of all 52 candidate sets, six new comparisons join one retained counting example;
  ambiguous or worse cases are not included merely to increase the count.
  Native formats include 1:1, 3:2, 2:3, 16:9 and 9:16. The original square examples
  retain their native 512/1024 resolution.
  No AI upscaling, cropping or retouching is applied to comparison images.
- The hero crops both genuine model outputs identically for presentation.
  The gallery comparisons show complete, unaltered frames.

## Publishing

GitHub Pages serves `main` from the repository root, without a build step or
external JavaScript/font requests. `.nojekyll` is retained. Serve with any
static HTTP server, or open `index.html` directly.

On an HPC cluster, image processing, inference and browser tests belong on
allocated compute nodes. This repository contains no weights or credentials.
