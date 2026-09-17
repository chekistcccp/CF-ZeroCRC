from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Img2ImgPipeline
from PIL import Image
from skimage.metrics import structural_similarity

from .postprocess import robust_normalize


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _calculate_shift(image_seq_len: int, base_seq_len: int, max_seq_len: int, base_shift: float, max_shift: float) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class SD35CounterfactualEngine:
    """Frozen SD3.5 engine for tri-prompt counterfactual flow and reconstruction maps."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.device = torch.device(model_cfg.get("device", "cuda"))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CF-ZeroCRC currently requires a CUDA GPU.")
        self.dtype = _dtype_from_name(model_cfg.get("dtype", "bfloat16"))
        self.model_path = Path(model_cfg["path"])
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run scripts/download_model.py first."
            )

        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.transformer.eval().requires_grad_(False)
        self.pipe.vae.eval().requires_grad_(False)

        self.prompt_bank = self._load_or_build_prompt_cache()
        self._move_generation_modules_to_gpu()

    def _prompt_cache_payload(self) -> dict[str, str]:
        return {
            "neutral": self.cfg["prompts"]["neutral"],
            "healthy": self.cfg["prompts"]["healthy"],
            "abnormal": self.cfg["prompts"]["abnormal"],
            "negative": "",
        }

    def _load_or_build_prompt_cache(self) -> dict[str, dict[str, torch.Tensor]]:
        cache_path = Path(self.cfg["model"].get("prompt_cache", "cache/sd35_prompt_embeddings.pt"))
        prompts = self._prompt_cache_payload()
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("prompts") == prompts:
                return payload["embeddings"]

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        for module_name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            module = getattr(self.pipe, module_name, None)
            if module is not None:
                module.to(self.device)

        bank: dict[str, dict[str, torch.Tensor]] = {}
        with torch.inference_mode():
            for key, text in prompts.items():
                prompt_embeds, _, pooled, _ = self.pipe.encode_prompt(
                    prompt=text,
                    prompt_2=text,
                    prompt_3=text,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                bank[key] = {
                    "prompt_embeds": prompt_embeds.detach().cpu(),
                    "pooled_prompt_embeds": pooled.detach().cpu(),
                }

        torch.save({"prompts": prompts, "embeddings": bank}, cache_path)

        for module_name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            module = getattr(self.pipe, module_name, None)
            if module is not None:
                module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return bank

    def _move_generation_modules_to_gpu(self) -> None:
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)
        torch.cuda.empty_cache()

    def _embeds(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.prompt_bank[key]
        return (
            item["prompt_embeds"].to(device=self.device, dtype=self.dtype),
            item["pooled_prompt_embeds"].to(device=self.device, dtype=self.dtype),
        )

    def _preprocess_image(self, image: Image.Image, resolution: int) -> torch.Tensor:
        x = self.pipe.image_processor.preprocess(image, height=resolution, width=resolution)
        return x.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def _encode_latent(self, image: Image.Image, resolution: int) -> torch.Tensor:
        x = self._preprocess_image(image, resolution)
        encoded = self.pipe.vae.encode(x)
        z0 = encoded.latent_dist.mode()
        z0 = (z0 - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return z0.to(dtype=self.dtype)

    def _set_scheduler(self, resolution: int, num_steps: int) -> None:
        kwargs: dict[str, Any] = {}
        if self.pipe.scheduler.config.get("use_dynamic_shifting", None):
            image_seq_len = (
                (resolution // self.pipe.vae_scale_factor // self.pipe.transformer.config.patch_size)
                * (resolution // self.pipe.vae_scale_factor // self.pipe.transformer.config.patch_size)
            )
            kwargs["mu"] = _calculate_shift(
                image_seq_len=image_seq_len,
                base_seq_len=self.pipe.scheduler.config.get("base_image_seq_len", 256),
                max_seq_len=self.pipe.scheduler.config.get("max_image_seq_len", 4096),
                base_shift=self.pipe.scheduler.config.get("base_shift", 0.5),
                max_shift=self.pipe.scheduler.config.get("max_shift", 1.16),
            )
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device, **kwargs)

    def _timestep_for_strength(self, strength: float, num_steps: int) -> torch.Tensor:
        strength = float(np.clip(strength, 1.0 / num_steps, 1.0))
        t_start = int(max(num_steps - num_steps * strength, 0))
        idx = min(t_start * self.pipe.scheduler.order, len(self.pipe.scheduler.timesteps) - 1)
        return self.pipe.scheduler.timesteps[idx]

    @torch.inference_mode()
    def flow_map(
        self,
        image: Image.Image,
        resolution: int,
        noise_levels: list[float],
        seed: int,
    ) -> np.ndarray:
        """Compute the multi-noise tri-prompt counterfactual flow map."""
        z0 = self._encode_latent(image, resolution)
        num_steps = int(self.cfg["flow"].get("scheduler_steps", 100))
        self._set_scheduler(resolution, num_steps)

        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        base_noise = torch.randn(z0.shape, generator=gen, device=self.device, dtype=self.dtype)
        maps: list[np.ndarray] = []

        e0, p0 = self._embeds("neutral")
        eh, ph = self._embeds("healthy")
        ea, pa = self._embeds("abnormal")

        for level in noise_levels:
            t = self._timestep_for_strength(float(level), num_steps)
            timestep = t.expand(z0.shape[0])
            zt = self.pipe.scheduler.scale_noise(z0, timestep, base_noise)

            def predict(emb: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
                return self.pipe.transformer(
                    hidden_states=zt,
                    timestep=timestep,
                    encoder_hidden_states=emb,
                    pooled_projections=pooled,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            v0 = predict(e0, p0)
            vh = predict(eh, ph)
            va = predict(ea, pa)

            d_h = torch.linalg.vector_norm(vh.float() - v0.float(), ord=2, dim=1)
            d_a = torch.linalg.vector_norm(va.float() - v0.float(), ord=2, dim=1)
            m = torch.relu(d_h - d_a).unsqueeze(1)
            m = F.interpolate(m, size=(resolution, resolution), mode="bilinear", align_corners=False)
            m_np = m[0, 0].detach().cpu().numpy().astype(np.float32)
            maps.append(robust_normalize(m_np))

        return robust_normalize(np.mean(maps, axis=0).astype(np.float32))

    @torch.inference_mode()
    def _reconstruct(self, image: Image.Image, key: str, resolution: int, seed: int) -> np.ndarray:
        fine_cfg = self.cfg["fine"]
        prompt, pooled = self._embeds(key)
        neg_prompt, neg_pooled = self._embeds("negative")
        gen = torch.Generator(device=self.device).manual_seed(int(seed))

        result = self.pipe(
            prompt=None,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            prompt_embeds=prompt,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_prompt,
            negative_pooled_prompt_embeds=neg_pooled,
            image=image,
            strength=float(fine_cfg["reconstruction_strength"]),
            num_inference_steps=int(fine_cfg["reconstruction_steps"]),
            guidance_scale=float(fine_cfg["guidance_scale"]),
            height=resolution,
            width=resolution,
            generator=gen,
            output_type="np",
        ).images[0]
        return np.asarray(result, dtype=np.float32).clip(0.0, 1.0)

    @staticmethod
    def _ssim_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        _, sim = structural_similarity(a, b, channel_axis=2, data_range=1.0, full=True)
        if sim.ndim == 3:
            sim = sim.mean(axis=2)
        return (1.0 - sim).astype(np.float32)

    @torch.inference_mode()
    def reconstruction_maps(self, image: Image.Image, resolution: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        neutral = self._reconstruct(image, "neutral", resolution, seed)
        healthy = self._reconstruct(image, "healthy", resolution, seed)
        abnormal = self._reconstruct(image, "abnormal", resolution, seed)

        r_h = np.mean(np.abs(healthy - neutral), axis=2)
        r_a = np.mean(np.abs(abnormal - neutral), axis=2)
        rec = np.maximum(r_h - r_a, 0.0)

        s_h = self._ssim_difference(healthy, neutral)
        s_a = self._ssim_difference(abnormal, neutral)
        ssim_map = np.maximum(s_h - s_a, 0.0)
        return robust_normalize(rec), robust_normalize(ssim_map)

    @torch.inference_mode()
    def fine_map(self, image: Image.Image) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        fine_cfg = self.cfg["fine"]
        fusion_cfg = self.cfg["fusion"]
        resolution = int(fine_cfg["resolution"])
        seeds = [int(s) for s in fine_cfg["seeds"]]
        noise_levels = [float(x) for x in self.cfg["flow"]["fine_noise_levels"]]
        use_rec = bool(fine_cfg.get("use_reconstruction", True))

        seed_maps: list[np.ndarray] = []
        flow_maps: list[np.ndarray] = []
        rec_maps: list[np.ndarray] = []
        ssim_maps: list[np.ndarray] = []

        for seed in seeds:
            flow = self.flow_map(image, resolution, noise_levels, seed)
            flow_maps.append(flow)

            if use_rec:
                rec, ssim_map = self.reconstruction_maps(image, resolution, seed)
            else:
                rec = np.zeros_like(flow)
                ssim_map = np.zeros_like(flow)
            rec_maps.append(rec)
            ssim_maps.append(ssim_map)

            wf = float(fusion_cfg.get("flow_weight", 0.5))
            wr = float(fusion_cfg.get("reconstruction_weight", 0.3)) if use_rec else 0.0
            ws = float(fusion_cfg.get("ssim_weight", 0.2)) if use_rec else 0.0
            total = max(wf + wr + ws, 1e-8)
            combined = (wf * flow + wr * rec + ws * ssim_map) / total
            seed_maps.append(robust_normalize(combined))

        stack = np.stack(seed_maps, axis=0).astype(np.float32)
        mean_map = stack.mean(axis=0)
        var_map = stack.var(axis=0)
        tau = max(float(fine_cfg.get("consistency_tau", 0.03)), 1e-6)
        consistency = np.exp(-var_map / tau).astype(np.float32)
        final = robust_normalize(mean_map * consistency)

        details = {
            "flow": np.mean(flow_maps, axis=0).astype(np.float32),
            "reconstruction": np.mean(rec_maps, axis=0).astype(np.float32),
            "ssim": np.mean(ssim_maps, axis=0).astype(np.float32),
            "consistency": consistency,
        }
        return final, details
