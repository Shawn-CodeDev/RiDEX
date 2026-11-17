# RiDEx — Reward-Informed Directional Exploration for Diffusion Reinforcement Learning

**Academic project website:** https://shawn-codedev.github.io/RiDEX/

This repository contains the **RiDEx project page** based on the research manuscript *RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning*.

The page includes the paper authors and affiliations, abstract, motivation, a three-stage method diagram, four-task H200 GPU-hour target-attainment comparisons, peak reward tables, local-credit ablations, and four original Base/RiDEx qualitative sample pairs.

## Activate GitHub Pages

This is a standalone static site: no build tools or external JavaScript packages are required.

**One-time GitHub setting:** open [Settings → Pages](https://github.com/Shawn-CodeDev/RiDEX/settings/pages), then under **Build and deployment** choose **Deploy from a branch**, **main**, **/(root)**, and **Save**. After publication, visit:

**https://shawn-codedev.github.io/RiDEX/**

Files are served from the root `index.html`; the site uses `assets/method-overview.svg`, four original web-optimized `assets/qualitative-*.webp` comparisons, and `favicon.svg`. The sample thumbnails are compressed crops of the manuscript's qualitative PDF, not newly generated examples.

## Key results

- **15×** HPSv2 target-attainment efficiency over DiffusionNFT
- **29×** PickScore target-attainment efficiency over DiffusionNFT
- Tests across GenEval, OCR, HPSv2, and PickScore on SD3.5 Medium

Speedups measure H200 GPU hours to a matched reward target.

## Manuscript and implementation

The website documents the uploaded research preprint. The public PDF identifier and complete training source will be linked separately when available; no non-existent PDF download URL is shown.

## Citation

```bibtex
@misc{yan2026ridex,
  title={RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning},
  author={Yan, Song and Wang, Chenfeng and Zhai, Wei and Wang, Xingchen and Zhang, Tao and Shao, Yawen and Pan, Shi and Li, Min and Zha, Zheng-Jun},
  year={2026},
  note={Preprint}
}
```
