from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from difix3d.model_ref import DifixRef


@dataclass
class DifixRefVanillaPipelineOutput:
    images: List[Image.Image]


def _to_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return Image.fromarray(img)
    raise TypeError(f"Unsupported image type: {type(img)}")


class DifixRefVanillaPipeline(DiffusionPipeline):
    """
    A minimal pipeline wrapper around `difix3d.model_ref.DifixRef`.

    - Loads the base model from `pretrained_name` (default: `nvidia/difix_ref`)
    - Optionally loads a finetuned checkpoint (`.pkl`) via `pretrained_path`
    - Accepts `image` (and optional `ref_image`) + `prompt` or pre-tokenized `prompt_tokens`

    This is intentionally simpler than Diffusers' StableDiffusion pipelines; it mirrors
    the repo's current `DifixRef.forward/sample` API.
    """

    def __init__(self, difix: DifixRef):
        super().__init__()
        self.register_modules(difix=difix)

    @property
    def device(self) -> torch.device:
        return next(self.difix.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: Optional[str] = "nvidia/difix_ref",
        pretrained_path: Optional[str] = None,
        *,
        lora_rank_vae: int = 4,
        timestep: int = 199,
        mv_unet: bool = False,
        deterministic_vae_encode: bool = True,
        deterministic_scheduler_step: bool = True,
    ) -> "DifixRefVanillaPipeline":
        difix = DifixRef(
            pretrained_name=pretrained_name,
            pretrained_path=pretrained_path,
            lora_rank_vae=lora_rank_vae,
            timestep=timestep,
            mv_unet=mv_unet,
            deterministic_vae_encode=deterministic_vae_encode,
            deterministic_scheduler_step=deterministic_scheduler_step,
        )
        difix.set_eval()
        return cls(difix=difix)

    def enable_xformers_memory_efficient_attention(self) -> None:
        if hasattr(self.difix, "unet") and hasattr(self.difix.unet, "enable_xformers_memory_efficient_attention"):
            self.difix.unet.enable_xformers_memory_efficient_attention()

    def _prompt_to_tokens(
        self,
        prompt: Optional[Union[str, Sequence[str]]],
        prompt_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (prompt is None) == (prompt_tokens is None):
            raise ValueError("Provide exactly one of `prompt` or `prompt_tokens`.")
        if prompt_tokens is not None:
            if not torch.is_tensor(prompt_tokens):
                raise TypeError("`prompt_tokens` must be a torch.Tensor.")
            return prompt_tokens.to(self.device)

        if isinstance(prompt, str):
            prompts: List[str] = [prompt]
        else:
            prompts = list(prompt)  # type: ignore[arg-type]
        tok = self.difix.tokenizer(
            prompts,
            max_length=self.difix.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tok.to(self.device)

    def _prepare_inputs(
        self,
        image: Union[Image.Image, np.ndarray],
        ref_image: Optional[Union[Image.Image, np.ndarray]],
        *,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        img = _to_pil(image).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)

        def to_tensor(p: Image.Image) -> torch.Tensor:
            arr = np.asarray(p).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
            t = t * 2.0 - 1.0
            return t

        if ref_image is None:
            x = to_tensor(img).unsqueeze(0).unsqueeze(0)  # (B=1,V=1,C,H,W)
        else:
            ref = _to_pil(ref_image).convert("RGB")
            ref = ref.resize((width, height), Image.LANCZOS)
            x = torch.stack([to_tensor(img), to_tensor(ref)], dim=0).unsqueeze(0)  # (1,2,C,H,W)

        return x.to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        *,
        ref_image: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        height: int = 112,
        width: int = 544,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[DifixRefVanillaPipelineOutput, List[Image.Image]]:
        if output_type not in {"pil"}:
            raise ValueError("Only `output_type='pil'` is supported in this minimal pipeline.")

        self.difix.set_eval()
        prompt_ids = self._prompt_to_tokens(prompt=prompt, prompt_tokens=prompt_tokens)

        # `DifixRef.forward` expects token ids; it will call text_encoder(prompt_tokens)
        dtype = next(self.difix.unet.parameters()).dtype if hasattr(self.difix, "unet") else torch.float32
        x = self._prepare_inputs(image, ref_image, height=height, width=width, dtype=dtype)

        out = self.difix(x, prompt_tokens=prompt_ids)  # (B,V,C,H,W) in [-1,1]
        out0 = out[0, 0].detach().float().cpu()  # (C,H,W)

        img = (out0 * 0.5 + 0.5).clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(img)

        images = [pil]
        if not return_dict:
            return images
        return DifixRefVanillaPipelineOutput(images=images)

