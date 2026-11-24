# HPSv2 training on 16 H200 GPUs

This release provides [RiDEx](https://arxiv.org/abs/2611.22052) and a clean-only
[DiffusionNFT](https://arxiv.org/abs/2509.16117) baseline using the same HPSv2.1
scorer and SD3.5 Medium backbone.

| Profile | Training GPUs | Prompt groups | Endpoints per group | Images per epoch |
| :--- | ---: | ---: | :--- | ---: |
| `ridex` | 16 H200 | 16 | 32 clean + 2 antithetic pairs | **576** |
| `diffusionnft` | 16 H200 | 48 | 24 clean | **1,152** |

The baseline uses the standard forward-process DiffusionNFT objective through
the shared trainer, not the upstream repository's launcher or its default
configuration. It does not use GenEval, paired rollouts, a differentiable
reward model on the training GPUs, or a directional update.

| Setting | Default |
| :--- | :--- |
| Backbone | SD3.5 Medium, LoRA rank 32 / alpha 64 |
| Rollout | 512 px, 10 DPM2 steps, CFG 1, FP32 |
| RiDEx exploration | 32 terminal HPSv2 action gradients per epoch, ninth velocity prediction, radius 0.005 |
| Update | One AdamW step per epoch, LR 3e-4. RiDEx adds the directional gradient to the NFT gradient |
| Evaluation | Every 10 updates, raw LoRA, 40 Euler steps, CFG 4.5 |
| Seed and duration | 42, 500 epochs |

## Setup

Use an isolated Linux environment with Python 3.12. Install PyTorch 2.12.0
and torchvision 0.27.0 for an index supported by your NVIDIA driver
([PyTorch wheel options](https://pytorch.org/get-started/previous-versions/)).
For example, with a CUDA 12.6-compatible driver:

```bash
cd training/hpsv2
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-train.txt
python -m pip check
bash submit.sh --dry-run ridex
bash submit.sh --dry-run diffusionnft
```

The scorer uses the `hpsv2` package, not `hpsv2x`.

Provide the following assets on a shared filesystem. They are not distributed
with this code.

- A local [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) model directory.
- [OpenCLIP ViT-H-14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K): place `open_clip_pytorch_model.bin` under `CLIP-ViT-H-14-laion2B-s32B-b79K/` in the reward model root.
- [HPSv2.1](https://huggingface.co/xswu/HPSv2/blob/main/HPS_v2.1_compressed.pt): place `HPS_v2.1_compressed.pt` under `HPSv2/` in the same root.
- A prompt directory containing UTF-8 `train.txt` and `test.txt`, one prompt per line and at least 48 nonempty training prompts to run both profiles. Use the original ordered splits for a run directly comparable to the paper. A different split changes the experiment; the bundled code does not fetch or replace the prompts.

## Train

Copy `site.env.example` to `site.env` and set the shared asset paths, Slurm
partition and local H200 GRES label (`H200_GRES_TYPE`). By default the launcher
requests **two nodes with eight H200s each** and a separate one-GPU scoring job.
Training checks the GPU model on every rank. Only RiDEx loads a frozen HPSv2
model on each training GPU for the action gradient. Use scheduler allocations
for training, scoring and validation. Authenticate with `wandb login` outside the
repo, or select `WANDB_MODE=offline` or `disabled`.

```bash
bash submit.sh --submit ridex
bash submit.sh --submit diffusionnft
```

For a first check, add `--stop-after-updates 1` to either command. This
completes a real epoch at the selected profile's image budget and saves one
checkpoint without running periodic evaluation.

Set `HPSV2_REWARD_URLS` in `site.env` to use a separately started
`python -m ridex.reward_server` instance. Use only this release's scorer
protocol. The service defaults to loopback; for multi-node use, bind it on
a trusted private network only. There is no built-in TLS or authentication.
For the automatic service, the service partition must have one GPU available
while training is allocated. Set a service time limit that covers training and
startup if you change the default time limits.

The trainer writes JSONL metrics, evaluation rows, and checkpoints under
`runs/`. `epoch` is zero-based and `completed_updates` is one-based. Reported
HPSv2 rewards are cosine x100, with raw cosine recorded separately.
Resume this release's online run with
`bash submit.sh --submit ridex --resume-run /shared/path/to/previous/output`
(select `diffusionnft` for that profile).
Keep the same allocation shape, prompt splits, and config, and stop any
previous writer before resuming.

## Verification and licensing

Run the CPU checks with `python -m unittest discover -s tests` in a
single-CPU scheduler allocation. They do not establish live GPU memory fit,
multi-node throughput, or reproduction of the published scores.

See [third-party notices](THIRD_PARTY_NOTICES.md) for component licenses.
The MixGRPO-derived sampling code has separate terms and is not MIT-licensed.
