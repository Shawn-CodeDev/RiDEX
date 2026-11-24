from pathlib import Path

import torch


class GradientReward:
    def __init__(self, model_root, device="cuda"):
        from flow_grpo.hpsv2_scorer import HPSv2Scorer

        root = Path(model_root)
        self.device = device
        self._text_cache = {}
        self.scorer = HPSv2Scorer(
            dtype=torch.float32,
            device=device,
            openclip_checkpoint=str(root / "CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"),
            hps_checkpoint=str(root / "HPSv2/HPS_v2.1_compressed.pt"),
        )
        self.model = self.scorer.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.scorer.eval()

    def score(self, images, prompts):
        if images.ndim != 4 or images.shape[1] not in (3, 4) or not torch.is_floating_point(images):
            raise ValueError("Expected floating-point NCHW RGB(A) images")
        if len(images) != len(prompts):
            raise ValueError("Expected one prompt per image")
        detached = images.detach()
        if not torch.isfinite(detached).all() or detached.min() < 0 or detached.max() > 1:
            raise ValueError("Expected finite image values in [0, 1]")
        image = self.scorer.preprocess_val(
            images.to(device=self.device, dtype=self.scorer.dtype, non_blocking=True)
        )
        features = self.model.encode_image(image, normalize=True)
        key = tuple(prompts)
        if key not in self._text_cache:
            tokens = self.scorer.processor(list(prompts)).to(device=self.device, non_blocking=True)
            with torch.no_grad():
                self._text_cache[key] = self.model.encode_text(tokens, normalize=True).detach()
        text = self._text_cache[key].to(device=features.device)
        return torch.diagonal(features @ text.T, 0).contiguous() * 100.
