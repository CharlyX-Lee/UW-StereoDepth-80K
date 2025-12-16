import numpy as np
import cv2
import PIL
from PIL import Image
import os
import json
import math
import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import torch.nn as nn
import skimage.io
import torch.nn.functional as F
import collections
import struct
import argparse
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline
from einops import rearrange, repeat
import copy
import torchvision
from diffusers.utils import load_image, export_to_video

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)
    return outputs


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.
    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """
    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)
            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0
            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])
        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
        return image_latents

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents
        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())
        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        print(frames.shape)
        num_frames = num_frames
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        temp_cond,
        mask,
        # pose,
        lambda_ts,
        lr,
        weight_clamp,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)
        num_frames = 25  # 强制设为25帧
        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        do_classifier_free_guidance = max_guidance_scale > 1.0
        # 3. Encode input image
        with torch.no_grad():
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        # NOTE: Stable Diffusion Video was conditioned on fps - 1
        fps = fps - 1
        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        mask = mask.cuda()
        mask = mask.unsqueeze(1).unsqueeze(0).repeat(1,1,4,1,1)
        temp_cond_list = []
        for i in range(len(temp_cond)):
            temp_cond_ = self.image_processor.preprocess(temp_cond[i], height=height, width=width)
            temp_cond_list.append(temp_cond_)
        temp_cond = torch.cat(temp_cond_list,dim=0)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        with torch.no_grad():
            image_latents = self._encode_vae_image(image[0:1,:,:,:], device, num_videos_per_prompt, do_classifier_free_guidance)
            temp_cond_latents_list = []
            for i in range(temp_cond.shape[0]):
                temp_cond_latents_ = self._encode_vae_image(temp_cond[i:i+1,:,:,:], device, num_videos_per_prompt, do_classifier_free_guidance)
                temp_cond_latents_ = rearrange(temp_cond_latents_, "(b f) c h w -> b f c h w",b=2)
                temp_cond_latents_list.append(temp_cond_latents_)
        temp_cond_latents = torch.cat(temp_cond_latents_list,dim=1)
        image_latents = rearrange(image_latents, "(b f) c h w -> b f c h w",f=1)
        temp_cond_latents  = torch.cat((image_latents,temp_cond_latents),dim=1)
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = image_latents.repeat(1, num_frames, 1, 1, 1)
        factor_s = 5.6
        temp_cond_latents = temp_cond_latents/factor_s
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)
        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale
        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                grads = []
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # 补充step_i=i参数（自定义调度器需要）
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)
                latent_model_input = torch.cat([latent_model_input[0:1], image_latents[1:2]], dim=2)
                for ii in range(4):
                    with torch.enable_grad(): 
                        latents.requires_grad_(True)
                        latents.retain_grad()
                        image_latents.requires_grad_(True)         
                        latent_model_input = latent_model_input.detach()
                        latent_model_input.requires_grad = True
                        # 冻结UNet参数
                        named_param = list(self.unet.named_parameters())
                        for n,p in named_param:
                            p.requires_grad = False
                        # 分块处理
                        if ii == 0:
                            latent_model_input1 = latent_model_input[0:1,:,:,:40,:72]
                            latents1 = latents[0:1,:,:,:40,:72]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,:72]
                            mask1 = mask[0:1,:,:,:40,:72]
                        elif ii ==1:
                            latent_model_input1 = latent_model_input[0:1,:,:,32:,:72]
                            latents1 = latents[0:1,:,:,32:,:72]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,32:,:72]
                            mask1 = mask[0:1,:,:,32:,:72]
                        elif ii ==2:
                            latent_model_input1 = latent_model_input[0:1,:,:,:40,56:]
                            latents1 = latents[0:1,:,:,:40,56:]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,56:]
                            mask1 = mask[0:1,:,:,:40,56:]
                        elif ii ==3:
                            latent_model_input1 = latent_model_input[0:1,:,:,32:,56:]
                            latents1 = latents[0:1,:,:,32:,56:]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,32:,56:]
                            mask1 = mask[0:1,:,:,32:,56:]
                        image_embeddings1 = image_embeddings[0:1,:,:]
                        added_time_ids1 = added_time_ids[0:1,:]
                        torch.cuda.empty_cache()
                        noise_pred_t = self.unet(
                            latent_model_input1,
                            t,
                            encoder_hidden_states=image_embeddings1,
                            added_time_ids=added_time_ids1,
                            return_dict=False,
                        )[0]
                        # 调用step_single（自定义调度器方法）
                        output = self.scheduler.step_single(
                            noise_pred_t, t, latents1, temp_cond_latents1, mask1,
                            lambda_ts, step_i=i, lr=lr, weight_clamp=weight_clamp, compute_grad=True
                        )
                        grad = output.grad
                        grads.append(grad)
                # 合并梯度
                grads1 = torch.cat((grads[0], grads[1][:,:,:,8:,:]), -2)                
                grads2 = torch.cat((grads[2], grads[3][:,:,:,8:,:]), -2)
                grads3 = torch.cat((grads1, grads2[:,:,:,:,16:]), -1)
                latents = latents - grads3.half()

                # 无梯度更新
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    # 补充step_i=i参数
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    # 调用step_single更新latents
                    latents = self.scheduler.step_single(
                        noise_pred, t, latents, temp_cond_latents, mask,
                        lambda_ts, step_i=i, compute_grad=False
                    ).prev_sample
                    # 回调函数
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                    # 更新进度条
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        # 解码并缩放回原始分辨率
        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            with torch.no_grad():
                frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return frames
        return StableVideoDiffusionPipelineOutput(frames=frames)


# resizing utils
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    # 计算sigma
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    # 计算核大小
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    # 确保核大小为奇数
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    # reshape for convolution
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    # convolve
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])
    return out


def forward_warp(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                    transformation1: np.ndarray, transformation2: np.ndarray, intrinsic1: np.ndarray,
                    intrinsic2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                np.ndarray]:
    """Warp frame1 to next view using bilinear splatting."""
    h, w = frame1.shape[:2]
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    # 断言输入形状
    assert frame1.shape == (h, w, 3)
    assert mask1.shape == (h, w)
    assert depth1.shape == (h, w)
    assert transformation1.shape == (4, 4)
    assert transformation2.shape == (4, 4)
    assert intrinsic1.shape == (3, 3)
    assert intrinsic2.shape == (3, 3)
    # 计算变换后的点
    trans_points1, world_points = compute_transformed_points(depth1, transformation1, transformation2, intrinsic1, intrinsic2)
    trans_coordinates = trans_points1[:, :, :2, 0] / (trans_points1[:, :, 2:3, 0])
    trans_depth1 = trans_points1[:, :, 2, 0]
    # 计算光流
    grid = create_grid(h, w)
    flow12 = trans_coordinates - grid
    # 双线性采样
    warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
    return warped_frame2, mask2, flow12


def compute_transformed_points(depth1: np.ndarray, transformation1: np.ndarray,
                                transformation2: np.ndarray, intrinsic1: np.ndarray,
                                intrinsic2: Optional[np.ndarray]):
    """Computes transformed position for each pixel location."""
    h, w = depth1.shape
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    # 计算变换矩阵
    transformation = np.matmul(transformation2, np.linalg.inv(transformation1))
    # 生成网格
    y1d = np.array(range(h))
    x1d = np.array(range(w))
    x2d, y2d = np.meshgrid(x1d, y1d)
    ones_2d = np.ones(shape=(h, w))
    ones_4d = ones_2d[:, :, None, None]
    pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]
    # 相机内参逆矩阵
    intrinsic1_inv = np.linalg.inv(intrinsic1)
    intrinsic1_inv_4d = intrinsic1_inv[None, None]
    intrinsic2_4d = intrinsic2[None, None]
    depth_4d = depth1[:, :, None, None]
    trans_4d = transformation[None, None]
    # 计算世界坐标
    unnormalized_pos = np.matmul(intrinsic1_inv_4d, pos_vectors_homo)
    world_points = depth_4d * unnormalized_pos
    world_points_homo = np.concatenate([world_points, ones_4d], axis=2)
    trans_world_homo = np.matmul(trans_4d, world_points_homo)
    trans_world = trans_world_homo[:, :, :3]
    trans_norm_points = np.matmul(intrinsic2_4d, trans_world)
    return trans_norm_points, world_points


def bilinear_splatting(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                        flow12: np.ndarray, flow12_mask: Optional[np.ndarray], is_image: bool = False) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Using inverse bilinear interpolation based splatting."""
    h, w, c = frame1.shape
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if flow12_mask is None:
        flow12_mask = np.ones(shape=(h, w), dtype=bool)
    # 生成网格
    grid = create_grid(h, w)
    trans_pos = flow12 + grid
    # 计算采样位置
    trans_pos_offset = trans_pos + 1
    trans_pos_floor = np.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
    # 边界裁剪
    trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_ceil[:, :, 0] = np.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_ceil[:, :, 1] = np.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)
    # 计算权重
    prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    # 深度权重
    sat_depth1 = np.clip(depth1, a_min=0, a_max=5000)
    log_depth1 = np.log(1 + sat_depth1)
    depth_weights = np.exp(log_depth1 / log_depth1.max() * 50)
    # 合并权重
    weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
    weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
    weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
    weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights
    # 3D权重
    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]
    # 初始化输出
    warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
    warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)
    # 加权求和
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)
    # 权重求和
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)
    # 裁剪到原始尺寸
    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]
    # 计算mask
    mask = cropped_weights > 0 
    mask2 = cropped_weights <= 0.6
    mask = mask * mask2
    # 归一化
    with np.errstate(invalid='ignore'):
        warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)
    # 图像类型转换
    if is_image:
        assert np.min(warped_frame2) >= 0
        assert np.max(warped_frame2) <= 256
        clipped_image = np.clip(warped_frame2, a_min=0, a_max=255)
        warped_frame2 = np.round(clipped_image).astype('uint8')
    return warped_frame2, mask


def create_grid(h, w):
    x_1d = np.arange(0, w)[None]
    y_1d = np.arange(0, h)[:, None]
    x_2d = np.repeat(x_1d, repeats=h, axis=0)
    y_2d = np.repeat(y_1d, repeats=w, axis=1)
    grid = np.stack([x_2d, y_2d], axis=2)
    return grid


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def look_at_matrix(camera_position, target, up):
    # Camera's forward vector (z-axis)
    forward = normalize(target - camera_position)
    # Camera's right vector (x-axis)
    right = normalize(np.cross(up, forward))
    # Camera's up vector (y-axis)
    up = np.cross(forward, right)
    # Rotation matrix
    rotation = np.array([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1]
    ])
    # Translation matrix
    translation = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])
    # View matrix
    view_matrix = rotation.T @ translation
    return view_matrix


def generate_camera_poses_around_ellipse(num_poses, angle_step, major_radius, minor_radius, inverse=False):
    """Generate camera poses around ellipse."""
    poses = []
    for i in range(num_poses):
        angle = np.deg2rad(angle_step * i if not inverse else 360 - angle_step * i)
        cam_x = major_radius * np.sin(angle)
        cam_z = minor_radius * np.cos(angle)
        look_at = np.array([0, 0, 0])  
        camera_position = np.array([cam_x, 0, cam_z])
        up_direction = np.array([0, 1, 0])  
        pose_matrix = look_at_matrix(camera_position, look_at, up_direction)
        poses.append(pose_matrix)
    return poses


def generate_camera_poses(start_position, end_position, num_poses):
    """Generate camera poses from start to end position."""
    poses = []
    cams = []
    t_values = np.linspace(0, 1, num_poses)
    for t in t_values:
        camera_position = (1 - t) * np.array(start_position) + t * np.array(end_position)
        look_at = np.array([0, 0, 0])  
        up_direction = np.array([0, 1, 0])  
        pose_matrix = look_at_matrix(camera_position, look_at, up_direction)
        poses.append(pose_matrix)
        cams.append(camera_position)
    return poses, cams


def save_warped_image(save_path, image_path, depth_path, num_frames, radius, end_position):
    # 模型标准分辨率（符合SVD要求）
    MODEL_WIDTH = 1024
    MODEL_HEIGHT = 576
    
    # 读取原始图像并记录分辨率
    original_image = PIL.Image.open(image_path)
    original_width, original_height = original_image.size
    
    # 调整图像到模型分辨率
    image_o = original_image.resize((MODEL_WIDTH, MODEL_HEIGHT), PIL.Image.Resampling.LANCZOS)
    image = np.array(image_o)
    
    # 加载并处理深度图
    depth = np.load(depth_path).astype(np.float32)
    depth[depth < 1e-5] = 1e-5
    depth = 10000. / depth 
    near, far = 0.0001, 500.
    depth = np.clip(depth, near, far)
    depth = cv2.resize(depth, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # 相机内参（基于模型分辨率）
    focal = 320.
    K = np.eye(3)
    K[0,0] = focal
    K[1,1] = focal
    K[0,2] = MODEL_WIDTH / 2.0
    K[1,2] = MODEL_HEIGHT / 2.0
    
    # 生成相机位姿
    start_position = [radius, 0, 0]
    poses, cams = generate_camera_poses(start_position, end_position, num_frames)
    
    pose_s = poses[0]
    cond_image = []
    masks = []
    
    # 生成warped图像和mask
    for i, pose_t in enumerate(poses[1:]):
        warped_frame2, mask2, flow12 = forward_warp(image, None, depth, pose_s, pose_t, K, None)
        
        # 处理mask
        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = np.repeat(mask[:,:,np.newaxis] * 255., repeats=3, axis=2)
        
        # 膨胀操作
        kernel = np.ones((5,5), np.uint8)
        mask_erosion = cv2.dilate(np.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))
        mask_erosion.save(os.path.join(save_path, f"{i:04d}_mask.png"))
        
        # 归一化mask
        mask_erosion_ = np.array(mask_erosion) / 255.
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        
        # 保存warped图像
        warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2))
        warped_frame2_masked = PIL.Image.fromarray(np.uint8(warped_frame2 * (1 - mask_erosion_)))
        warped_frame2_masked.save(os.path.join(save_path, f"{i:04d}.png"))
        cond_image.append(warped_frame2_masked)
        
        # 调整mask到潜在空间尺寸（VAE下采样8倍）
        h, w = mask_erosion_.shape[:2]
        latent_h, latent_w = h // 8, w // 8
        mask_erosion_gray = np.mean(mask_erosion_, axis=-1)
        mask_erosion_resized = cv2.resize(mask_erosion_gray, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
        mask_erosion_resized[mask_erosion_resized < 0.2] = 0
        mask_erosion_resized[mask_erosion_resized >= 0.2] = 1
        masks.append(torch.from_numpy(mask_erosion_resized).unsqueeze(0))
    
    masks = torch.cat(masks)
    # 返回所有必要参数
    return original_image, image_o, masks, cond_image, original_height, original_width, MODEL_HEIGHT, MODEL_WIDTH


def svd_render(original_image, model_image, masks, cond_image, image_path, warped_img_path, output_path, 
               lambda_ts, lr, weight_clamp, original_height, original_width, model_height, model_width):
    # 使用【自定义】的StableVideoDiffusionPipeline（而非标准版）
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    pipe.to("cuda")
    
    # 调用管道时传入所有必填参数（temp_cond、mask等）
    frames = pipe(
        [model_image],  # 模型分辨率图像（符合SVD要求）
        temp_cond=cond_image,  # save_warped_image生成的条件图像
        mask=masks,  # save_warped_image生成的mask
        lambda_ts=lambda_ts,  # 超参数
        lr=lr,  # 学习率
        weight_clamp=weight_clamp,  # 权重裁剪
        height=model_height,
        width=model_width,
        num_frames=25,
        num_inference_steps=100,
        decode_chunk_size=8,
        output_type="pil"
    ).frames[0]
    
    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 将生成的帧【缩放回原始分辨率】
    resized_frames = []
    for i, fr in enumerate(frames):
        # 使用LANCZOS插值保证缩放质量
        resized_frame = fr.resize((original_width, original_height), PIL.Image.Resampling.LANCZOS)
        resized_frame.save(os.path.join(output_path, f"{i:06d}.png"))
        resized_frames.append(resized_frame)
    
    # 导出原始分辨率的视频
    export_to_video(resized_frames, os.path.join(output_path, "generated.mp4"), fps=7)
    return resized_frames


def search_hypers(sigmas, save_path):
    sigmas = sigmas[:-1]
    sigmas_max = max(sigmas)
    v2_list = np.arange(50, 1001, 50)
    v3_list = np.arange(10, 101, 10)
    v1_list = np.linspace(0.001, 0.009, 9)
    zero_count_default = 0
    index_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    for v1 in v1_list:
        for v2 in v2_list:
            for v3 in v3_list:
                flag = True
                lambda_t_list = []
                for sigma in sigmas:
                    sigma_n = sigma/sigmas_max
                    temp_cond_indices = [0]
                    for tau in range(25):
                        if tau not in index_list:
                            lambda_t_list.append(1)
                        else:
                            tau_p = 0 
                            tau_ = tau/24
                            Q = v3 * abs((tau_-tau_p)) - v2*sigma_n
                            k = 0.8
                            b = -0.2
                            # 计算lambda_t的四个解
                            lambda_t_1 = (-(2*v1 + k*Q) + ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                            lambda_t_2 = (-(2*v1 + k*Q) - ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                            v1_ = -v1
                            lambda_t_3 = (-(2*v1_ + k*Q) + ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                            lambda_t_4 = (-(2*v1_ + k*Q) - ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                            try:
                                # 选择有效解
                                if np.isreal(lambda_t_1) and lambda_t_1 > 1.0:
                                    lambda_t_list.append(lambda_t_1/(1+lambda_t_1))
                                    continue
                                if np.isreal(lambda_t_2) and lambda_t_2 > 1.0:
                                    lambda_t_list.append(lambda_t_2/(1+lambda_t_2))
                                    continue
                                if np.isreal(lambda_t_3) and 0 < lambda_t_3 <= 1.0:
                                    lambda_t_list.append(lambda_t_3/(1+lambda_t_3))
                                    continue
                                if np.isreal(lambda_t_4) and 0 < lambda_t_4 <= 1.0:
                                    lambda_t_list.append(lambda_t_4/(1+lambda_t_4))
                                    continue
                                flag = False
                                break
                            except:
                                flag = False
                                break
                    if not flag:
                        break
                # 更新最优参数
                if flag:
                    zero_count = sum(1 for x in lambda_t_list if x > 0.5)
                    if zero_count > zero_count_default:
                        zero_count_default = zero_count
                        v_optimized = [v1, v2, v3]
                        lambda_t_list_optimized = lambda_t_list
    
    # 可视化并保存lambda_t
    X = np.array(sigmas)
    Y = np.arange(0,25,1)
    X, Y = np.meshgrid(X, Y)
    lambda_t_list_optimized = np.array(lambda_t_list_optimized).reshape([len(sigmas),25])
    lambda_t_list_optimized = torch.tensor(lambda_t_list_optimized)
    Z = lambda_t_list_optimized
    z_upsampled = F.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='bilinear', align_corners=True)
    save_path = os.path.join(save_path, f'lambad_{v_optimized[0]}_{v_optimized[1]}_{v_optimized[2]}.png')
    image_numpy = z_upsampled[0].permute(1, 2, 0).numpy()
    plt.figure()  
    plt.imshow(image_numpy)  
    plt.colorbar()  
    plt.axis('off')  
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close() 
    return lambda_t_list_optimized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, default=50)
    parser.add_argument('--end_position', nargs=3, type=float, required=True)
    parser.add_argument('--lr', type=float, default=10)
    parser.add_argument('--weight_clamp', type=float, default=0.4)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--folder_path', type=str)
    parser.add_argument('--iteration', type=str)
    args = parser.parse_args()
    
    num_frames = 25  # 固定生成25帧
    # 创建保存目录
    save_path = os.path.join(args.folder_path, 'warp_' + args.iteration)
    output_path = os.path.join(args.folder_path, 'render_warp_' + args.iteration)
    image_name, _ = os.path.splitext(os.path.basename(args.image_path))
    depth_path = os.path.join(args.folder_path, 'depth', image_name + '.npy')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 生成warped图像、mask和条件图像
    original_image, model_image, masks, cond_image, original_height, original_width, model_height, model_width = save_warped_image(
        save_path, args.image_path, depth_path, num_frames, args.radius, args.end_position
    )
    
    print(f"原始分辨率: {original_width}x{original_height}")
    print(f"模型分辨率: {model_width}x{model_height}")
    
    # 加载sigma并搜索最优lambda_ts（替换之前的默认值）
    sigma_list = np.load('sigmas/sigmas_100.npy').tolist()
    lambda_ts = search_hypers(sigma_list, args.folder_path)
    
    # 渲染并缩放回原始分辨率
    frames = svd_render(
        original_image, model_image, masks, cond_image, args.image_path, save_path, output_path, 
        lambda_ts, args.lr, args.weight_clamp, original_height, original_width, model_height, model_width
    )
    
    print(f"生成完成！视频已保存到: {output_path}")