from dataclasses import asdict, dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model: str
    reward_models: str
    dataset: str
    method: str = "RiDEx"
    prompts_per_epoch: int = 16
    clean_per_prompt: int = 32
    centers_per_prompt: int = 2
    action_step: int = 8
    action_radius: float = .005
    resolution: int = 512
    steps: int = 10
    learning_rate: float = 3e-4
    train_microbatch: int = 1
    sequence_length: int = 128
    eval_steps: int = 40
    eval_guidance: float = 4.5
    save_freq: int = 10
    eval_freq: int = 10
    seed: int = 42
    epochs: int = 500

    @property
    def reward(self):
        return "hpsv2"

    @property
    def proposal(self):
        return "terminal_gradient" if self.centers_per_prompt else None

    @property
    def reward_weights(self):
        return {"hpsv2": 1.}

    @property
    def display_scale(self):
        return 100.

    def validate(self, world_size=1):
        for name in ("prompts_per_epoch", "clean_per_prompt", "centers_per_prompt", "action_step",
                     "resolution", "steps", "train_microbatch", "sequence_length", "eval_steps",
                     "save_freq", "eval_freq", "seed", "epochs"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= world_size <= self.prompts_per_epoch:
            raise ValueError("Each GPU must own at least one complete prompt group")
        if self.method not in ("RiDEx", "DiffusionNFT"):
            raise ValueError("Expected RiDEx or DiffusionNFT")
        if self.method == "DiffusionNFT" and self.centers_per_prompt != 0:
            raise ValueError("DiffusionNFT cannot use directional pairs")
        if self.clean_per_prompt < 1:
            raise ValueError("Expected at least one clean sample per prompt")
        if self.method == "RiDEx" and not 1 <= self.centers_per_prompt <= self.clean_per_prompt:
            raise ValueError("Invalid center budget")
        if not 0 <= self.action_step < self.steps:
            raise ValueError("action_step is a zero-based velocity call index")
        if not math.isfinite(self.action_radius) or not 0 < self.action_radius < 1:
            raise ValueError("Invalid relative action radius")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if min(self.epochs, self.save_freq, self.eval_freq) < 1 or not 0 <= self.seed < 2**31:
            raise ValueError("Invalid epochs/cadence/seed")
        if (self.steps, self.resolution, self.train_microbatch) != (10, 512, 1):
            raise ValueError("This profile uses ten calls, 512 pixels, microbatch one")
        if self.eval_steps < 1 or self.sequence_length < 1 or not math.isfinite(self.eval_guidance) or self.eval_guidance <= 1:
            raise ValueError("Invalid evaluation or text-conditioning settings")
        for split in ("train", "test"):
            path = Path(self.dataset) / f"{split}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)

    def budget(self):
        clean = self.prompts_per_epoch * self.clean_per_prompt
        pairs = self.prompts_per_epoch * self.centers_per_prompt
        return dict(clean_images=clean, pair_images=2*pairs, total_images=clean+2*pairs,
            direction_backward_calls=pairs, canonical_scorer_images=clean+2*pairs)

    def phase_end_update(self, requested):
        completed = self.epochs if requested is None else requested
        if type(completed) is not int or not 1 <= completed <= self.epochs:
            raise ValueError("Phase end must be a positive update count within the planned experiment")
        return completed

    def to_dict(self):
        return dict(asdict(self), version="ridex_hpsv2", reward=self.reward,
            **self.budget(), direction_source=self.proposal, display_scale=self.display_scale,
            reward_weights=self.reward_weights, gradient_reward=self.reward if self.proposal else None,
            exploration_axes=[self.proposal] if self.proposal else [],
            exploration_coordinates="single_axis" if self.proposal else None,
            rollout_precision="float32_tf32_off", training_precision="float32",
            text_encoder_precision="float16_frozen", conditioning_precision="float32",
            rollout_guidance=1., rollout_solver="dpm2", optimizer_updates_epoch=1,
            implicit_nft_beta=1., kl_weight=1e-4, advantage_clip=5.,
            local_loss_weight=1. if self.proposal else 0.,
            gradient_clip_norm=1., lora_rank=32, lora_alpha=64, ema=False,
            eval_protocol="fixed_row_seed_arithmetic_v27", eval_solver="flow_euler",
            eval_generator_device="cpu", eval_weights="raw", sample_filter=False,
            uncertainty_shrinkage=False, update_rule="joint_adam" if self.proposal else "nft_adam",
            optimizer_moments="joint" if self.proposal else "nft_only",
            terminal_credit="canonical_scalar_pair_difference" if self.proposal else None,
            action_units="velocity_l2_relative" if self.proposal else None,
            canonical_training_units="cosine")
