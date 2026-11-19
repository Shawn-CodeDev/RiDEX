<h1 align="center">RiDEx</h1>
<p align="center"><strong>Reward-Informed Directional Exploration for Diffusion Reinforcement Learning</strong></p>

<p align="center">
  Song Yan<sup>1</sup> · Chenfeng Wang<sup>1,2</sup> · Wei Zhai<sup>1,†</sup> · Xingchen Wang<sup>1</sup> · Tao Zhang<sup>3</sup><br>
  Yawen Shao<sup>1,4</sup> · Shi Pan<sup>5</sup> · Min Li<sup>5,†</sup> · Zheng-Jun Zha<sup>1</sup>
</p>
<p align="center">
  <sup>1</sup> University of Science and Technology of China · <sup>2</sup> Li Auto Inc.<br>
  <sup>3</sup> Huazhong University of Science and Technology · <sup>4</sup> Tongyi Lab<br>
  <sup>5</sup> Xi'an High-tech Research Institute<br>
  <em>† Corresponding authors</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.22052"><img alt="Read the RiDEx paper on arXiv" src="https://img.shields.io/badge/arXiv-2611.22052-B33A40?logo=arxiv&amp;logoColor=white"></a>
  <a href="https://gary.ac.cn/RiDEX/"><img alt="Visit the RiDEx project page" src="https://img.shields.io/badge/Project_Page-36715E?logo=github&amp;logoColor=white"></a>
  <a href="https://gary.ac.cn/RiDEX/#comparisons"><img alt="Explore the visual comparisons" src="https://img.shields.io/badge/Visual_Gallery-536EA2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB4PSIzIiB5PSIzIiB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHJ4PSIyIi8%2BPGNpcmNsZSBjeD0iOSIgY3k9IjgiIHI9IjEiLz48cGF0aCBkPSJtMyAxNyA1LTUgNCA0IDQtNiA1IDciLz48L3N2Zz4%3D"></a>
  <a href="#citation"><img alt="Get the BibTeX citation" src="https://img.shields.io/badge/BibTeX-705783?logo=latex&amp;logoColor=white"></a>
  <a href="LICENSE"><img alt="Read the MIT License" src="https://img.shields.io/badge/License-MIT-596472"></a>
</p>

**Which changes to a denoising step actually matter to the reward?** RiDEx starts two trajectories from the same diffusion state and nudges their velocities in opposite directions. Comparing the finished images tells the model which way to move at that state. The reward judges the images, not the proposed direction.

**Release status:** The paper, training curves and 40 matched image comparisons are available now. The training implementation and runnable configurations are not yet public.

## A matched comparison

*Prompt excerpt: “exactly two red apples and one green pear.”* The three models receive the same complete prompt and seed. Base produces two fruits; DiffusionNFT produces three but swaps an apple for a pear. RiDEx gets both counts right in this example.

| Base · SD3.5 Medium | DiffusionNFT | RiDEx |
| :---: | :---: | :---: |
| ![Base: one red apple and one green fruit](assets/examples/expanded_g_apples_pears_base.webp) | ![DiffusionNFT: one red apple and two green pears](assets/examples/expanded_g_apples_pears_diffusionnft.webp) | ![RiDEx: two red apples and one green pear](assets/examples/expanded_g_apples_pears_ridex.webp) |

The [gallery](https://gary.ac.cn/RiDEX/#comparisons) contains ten selected prompts each for GenEval, OCR, HPSv2 and PickScore. Open any image to compare methods at native resolution. The [manifest](assets/comparison_manifest.json) records prompts, seeds and checkpoints; these examples are qualitative, not a preference win rate.

## How RiDEx learns

1. **Propose a direction.** Use a conditional residual or a differentiable reward gradient to choose a small set of velocity directions worth testing.
2. **Test it at the same state.** Apply opposite perturbations, finish both rollouts, and compare their terminal rewards. Either sign can win; a tie provides no directional preference.
3. **Learn at two levels.** Use the signed difference to supervise the tested velocity direction at its source state. Use all generated images for the global DiffusionNFT update.

## Training results

[![Evaluation reward against H200 GPU hours for RiDEx and DiffusionNFT on GenEval, OCR, HPSv2 and PickScore](assets/training_comparison.webp)](assets/training_comparison.pdf)

On SD3.5 Medium, the [paper](https://arxiv.org/abs/2611.22052) reports that RiDEx needs **1/15 of DiffusionNFT's H200 GPU hours on HPSv2** and **1/29 on PickScore** to reach the respective target scores. This compares compute to reach a score, not the speed of a training step. [View the vector figure (PDF)](assets/training_comparison.pdf).

## Citation

```bibtex
@misc{yan2026ridexrewardinformeddirectionalexploration,
  title={RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning},
  author={Song Yan and Chenfeng Wang and Wei Zhai and Xingchen Wang and Tao Zhang and Yawen Shao and Shi Pan and Min Li and Zheng-Jun Zha},
  year={2026},
  eprint={2611.22052},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.22052},
}
```

## License

RiDEx code is available under the [MIT License](LICENSE). The bundled fonts retain their respective [OFL notices](assets/fonts/).

[Project page source and local preview](web/README.md)
