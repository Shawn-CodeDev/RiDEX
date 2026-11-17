# RiDEx — Reward-Informed Directional Exploration

Project page for **RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning**.

**Website:** https://shawn-codedev.github.io/RiDEX/

## Website design

The site uses a fullscreen mosaic of original RiDEx generations, a three-stage method visualization, task-switchable H200 GPU-hour comparisons, sortable peak-reward results, a four-task credit ablation, eight draggable Base/RiDEx image comparisons, and the manuscript abstract and citation. Its visual direction is inspired by the DiffusionOPSD research website; the code and layout are written for RiDEx.

`index.html`, `styles.css`, and `script.js` are standalone static files, with no external JavaScript or font dependencies. Four AVIF sprite sheets contain compressed crops from the author's qualitative PDF plates. Each row is a genuine Base/RiDEx pair; no new samples are generated for the site. Previous assets are retained in the repository.

## GitHub Pages

Publish the **main** branch from **/(root)** in Settings → Pages. The `.nojekyll` file supports static serving. For local preview, run `python -m http.server 8000` from the repository and visit http://localhost:8000/.

## Research content

The text, author list, task mappings, and numerical results follow the supplied arXiv-source manuscript. CFG is framed by text–image alignment objectives; Grad is framed by differentiable reward paths. The 15× and 29× comparisons measure H200 GPU hours to a matched DiffusionNFT reward target, not generation latency. A public manuscript URL can be added when available; the site does not link to a nonexistent PDF.

## Accessibility

Responsive layouts, keyboard-operable task tabs and comparison sliders, an accessible mobile menu, a skip link, reduced-motion support, a motion toggle, horizontal table scrolling, and BibTeX copy with a fallback are included.
