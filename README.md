# RiDEx project page

[Live website](https://gary.ac.cn/RiDEX/)

A buildless research page: an editorial artwork hero, the abstract, the supplied
four-task training comparison PDF, matched Base / DiffusionNFT / RiDEx image
comparisons, and a citation.
No vector method explainer, duplicated result tables, or frontend framework.
Page sequence: hero, abstract, results, generations, citation.

## Hero

`assets/hero-art.webp` is decorative editorial artwork, not a RiDEx or baseline
model output and not evidence for the method. The paper title, full author list,
and affiliations appear over the artwork's open upper field. `hero.css` owns the
hero presentation; `navigation.js` owns the responsive navigation and scroll
header. Inter and italic Playfair Display are self-hosted with their OFL licenses.

## Assets

- `assets/training_comparison.pdf` is the supplied paper figure, unchanged.
  The WebP preview is rendered directly from the PDF without redrawing data.
- Individual native-resolution images replace compressed paper-plate crops.
  Every triplet matches prompt, seed, resolution, 40-step Euler sampler and CFG
  4.5 across methods. These curated examples illustrate selected outcomes; they
  do not establish universal wins. Click an image to inspect it and switch methods.
- `assets/comparison_manifest.json` records settings and checkpoint identities.
  Examples are selected, not aggregate win-rate evidence.
- The gallery includes 10 matched examples per reward task, 40 triplets / 120 images,
  with square, landscape and portrait filters and a clickable image wall. Examples
  are selected for inspection, not as aggregate performance evidence.
  Native formats include 1:1, 3:2, 2:3, 16:9 and 9:16. Newly generated comparisons use
  1152x768, 768x1152 and 1280x720 frames; retained cases keep their original pixels.
  No AI upscaling, cropping or retouching is applied to comparison images.
- Gallery comparisons show complete, unaltered frames; the decorative hero art
  is separate from the experiment image set.

## Publishing

GitHub Pages serves `main` from the repository root, without a build step or
external JavaScript/font requests. `.nojekyll` is retained. Serve with any
static HTTP server, or open `index.html` directly.

On an HPC cluster, image processing, inference and browser tests belong on
allocated compute nodes. This repository contains no weights or credentials.
