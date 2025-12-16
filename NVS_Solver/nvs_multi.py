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
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import torch.nn as nn
import skimage.io
import torch.nn.functional as F
import collections
import struct
import argparse
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Union
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
import shutil  # 新增：用于复制深度图

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# ===================== 分布式初始化函数 =====================
def init_distributed(args):
    """初始化分布式环境（单卡直接跳过，多卡自动设置环境变量）"""
    # ========== 单卡场景：跳过分布式，简化逻辑 ==========
    if torch.cuda.device_count() == 1:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.is_main_process = True  # 单卡默认为主进程
        print(f"✅ 单卡模式，跳过分布式初始化 | 使用设备: {args.device}")
        return
    
    # ========== 多卡场景（mp.spawn启动）：自动设置分布式环境变量 ==========
    # 手动设置主节点地址和端口（避免环境变量缺失）
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")  # 选一个未被占用的端口（如29500/29501）
    
    # 从mp.spawn传入的proc_id设置rank
    args.rank = args.proc_id
    args.local_rank = args.proc_id
    args.world_size = torch.cuda.device_count()
    
    # 绑定当前进程到指定GPU
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f"cuda:{args.local_rank}")
    
    # 初始化分布式进程组（GPU用NCCL后端）
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank
        )
        args.is_main_process = (args.rank == 0)
        if args.is_main_process:
            print(f"✅ 多卡分布式初始化完成 | 总卡数: {args.world_size} | 当前卡rank: {args.rank}")
    except Exception as e:
        raise RuntimeError(f"❌ 多卡分布式初始化失败: {e}")

# ===================== 原有工具函数（仅适配device） =====================
def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)
    return outputs

@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]

class StableVideoDiffusionPipeline(DiffusionPipeline):
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
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0
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
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
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
            image_latents = torch.cat([negative_image_latents, image_latents])
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
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents
        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
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
                f" size of {batch_size}."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def __call__(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
            temp_cond,
            mask,
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
            device: torch.device = None,  # 新增：支持指定device
        ):
        # ========== 核心修改1：强制指定device，兜底为cuda:0 ==========
        if device is None:
            # 优先用pipeline的device，否则强制cuda:0
            device = getattr(self, 'device', torch.device('cuda:0')) if hasattr(self, 'device') else torch.device('cuda:0')
        self.to(device)  # 确保pipeline本身在目标设备
    
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        self.check_inputs(image, height, width)
        num_frames = 25
        batch_size = 1
        do_classifier_free_guidance = max_guidance_scale > 1.0
    
        with torch.no_grad():
            # ========== 核心修改2：确保image_embeddings在目标设备 ==========
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
            image_embeddings = image_embeddings.to(device)
    
        fps = fps - 1
        # ========== 核心修改3：preprocess后的image强制移到device ==========
        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        # ========== 核心修改4：mask确保在device且dtype匹配 ==========
        mask = mask.to(device, dtype=image.dtype)
        mask = mask.unsqueeze(1).unsqueeze(0).repeat(1,1,4,1,1)
        
        # ========== 核心修改5：temp_cond每个元素都移到device后再拼接 ==========
        temp_cond_list = []
        for i in range(len(temp_cond)):
            temp_cond_ = self.image_processor.preprocess(temp_cond[i], height=height, width=width).to(device)
            temp_cond_list.append(temp_cond_)
        temp_cond = torch.cat(temp_cond_list,dim=0).to(device)
        
        # ========== 核心修改6：noise生成时直接指定device，dtype匹配image ==========
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        # ========== 核心修改7：noise_aug_strength转为张量并移到device ==========
        noise_aug_strength = torch.tensor(noise_aug_strength, dtype=image.dtype, device=device)
        image = image + noise_aug_strength * noise  # 此时所有张量都在device
    
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        
        with torch.no_grad():
            # ========== 核心修改8：_encode_vae_image输入确保在device ==========
            image_latents = self._encode_vae_image(image[0:1,:,:,:], device, num_videos_per_prompt, do_classifier_free_guidance)
            image_latents = image_latents.to(device)
            
            temp_cond_latents_list = []
            for i in range(temp_cond.shape[0]):
                temp_cond_latents_ = self._encode_vae_image(temp_cond[i:i+1,:,:,:], device, num_videos_per_prompt, do_classifier_free_guidance)
                temp_cond_latents_ = rearrange(temp_cond_latents_, "(b f) c h w -> b f c h w",b=2).to(device)
                temp_cond_latents_list.append(temp_cond_latents_)
        
        # ========== 核心修改9：拼接后强制移到device ==========
        temp_cond_latents = torch.cat(temp_cond_latents_list,dim=1).to(device)
        image_latents = rearrange(image_latents, "(b f) c h w -> b f c h w",f=1).to(device)
        temp_cond_latents  = torch.cat((image_latents,temp_cond_latents),dim=1).to(device)
        
        # ========== 核心修改10：dtype转换时保留device ==========
        image_latents = image_latents.to(image_embeddings.dtype).to(device)
        image_latents = image_latents.repeat(1, num_frames, 1, 1, 1).to(device)
        factor_s = 5.6
        temp_cond_latents = (temp_cond_latents/factor_s).to(device)
        
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    
        # ========== 核心修改11：added_time_ids生成后移到device并匹配dtype ==========
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength.item(),  # 转回标量避免张量维度问题
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device, dtype=image_embeddings.dtype)
    
        # ========== 核心修改12：scheduler的timesteps强制移到device ==========
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)
    
        num_channels_latents = self.unet.config.in_channels
        # ========== 核心修改13：prepare_latents返回后强制移到device ==========
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
        ).to(device)
    
        # ========== 核心修改14：guidance_scale生成时直接指定device ==========
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames, device=device).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale
    
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                grads = []
                # ========== 核心修改15：latent_model_input拼接后移到device ==========
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)
                latent_model_input = torch.cat([latent_model_input[0:1], image_latents[1:2]], dim=2).to(device)
                
                for ii in range(4):
                    with torch.enable_grad(): 
                        latents.requires_grad_(True)
                        latents.retain_grad()
                        image_latents.requires_grad_(True)         
                        latent_model_input = latent_model_input.detach()
                        latent_model_input.requires_grad = True
                        
                        named_param = list(self.unet.named_parameters())
                        for n,p in named_param:
                            p.requires_grad = False
                        
                        # ========== 核心修改16：切片后的张量强制保留device ==========
                        if ii == 0:
                            latent_model_input1 = latent_model_input[0:1,:,:,:40,:72].to(device)
                            latents1 = latents[0:1,:,:,:40,:72].to(device)
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,:72].to(device)
                            mask1 = mask[0:1,:,:,:40,:72].to(device)
                        elif ii ==1:
                            latent_model_input1 = latent_model_input[0:1,:,:,32:,:72].to(device)
                            latents1 = latents[0:1,:,:,32:,:72].to(device)
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,32:,:72].to(device)
                            mask1 = mask[0:1,:,:,32:,:72].to(device)
                        elif ii ==2:
                            latent_model_input1 = latent_model_input[0:1,:,:,:40,56:].to(device)
                            latents1 = latents[0:1,:,:,:40,56:].to(device)
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,56:].to(device)
                            mask1 = mask[0:1,:,:,:40,56:].to(device)
                        elif ii ==3:
                            latent_model_input1 = latent_model_input[0:1,:,:,32:,56:].to(device)
                            latents1 = latents[0:1,:,:,32:,56:].to(device)
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,32:,56:].to(device)
                            mask1 = mask[0:1,:,:,32:,56:].to(device)
                        
                        # ========== 核心修改17：子张量移到device ==========
                        image_embeddings1 = image_embeddings[0:1,:,:].to(device)
                        added_time_ids1 = added_time_ids[0:1,:].to(device)
                        torch.cuda.empty_cache()
                        
                        noise_pred_t = self.unet(
                            latent_model_input1,
                            t,
                            encoder_hidden_states=image_embeddings1,
                            added_time_ids=added_time_ids1,
                            return_dict=False,
                        )[0]
                        noise_pred_t = noise_pred_t.to(device)  # 确保输出在device
                        
                        output = self.scheduler.step_single(
                            noise_pred_t, t, latents1, temp_cond_latents1, mask1,
                            lambda_ts, step_i=i, lr=lr, weight_clamp=weight_clamp, compute_grad=True
                        )
                        grad = output.grad.to(device)  # 确保梯度在device
                        grads.append(grad)
                
                # ========== 核心修改18：拼接梯度时强制移到device ==========
                grads1 = torch.cat((grads[0], grads[1][:,:,:,8:,:]), -2).to(device)                
                grads2 = torch.cat((grads[2], grads[3][:,:,:,8:,:]), -2).to(device)
                grads3 = torch.cat((grads1, grads2[:,:,:,:,16:]), -1).to(device)
                # ========== 核心修改19：half()转换前确保在device ==========
                latents = latents - grads3.half().to(device)
    
                with torch.no_grad():
                    # ========== 核心修改20：重新拼接的latent_model_input移到device ==========
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2).to(device)
                    
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.to(device)  # 确保输出在device
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    latents = self.scheduler.step_single(
                        noise_pred, t, latents, temp_cond_latents, mask,
                        lambda_ts, step_i=i, compute_grad=False
                    ).prev_sample.to(device)  # 确保prev_sample在device
                    
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k].to(device) if isinstance(locals()[k], torch.Tensor) else locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents).to(device)
                    
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
    
        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            with torch.no_grad():
                # ========== 核心修改21：decode_latents输入确保在device ==========
                frames = self.decode_latents(latents.to(device), num_frames, decode_chunk_size)
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
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def _compute_padding(kernel_size):
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
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
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
                    intrinsic2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = frame1.shape[:2]
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    assert frame1.shape == (h, w, 3)
    assert mask1.shape == (h, w)
    assert depth1.shape == (h, w)
    assert transformation1.shape == (4, 4)
    assert transformation2.shape == (4, 4)
    assert intrinsic1.shape == (3, 3)
    assert intrinsic2.shape == (3, 3)
    
    trans_points1, world_points = compute_transformed_points(depth1, transformation1, transformation2, intrinsic1, intrinsic2)
    trans_coordinates = trans_points1[:, :, :2, 0] / (trans_points1[:, :, 2:3, 0] + 1e-8)
    trans_depth1 = trans_points1[:, :, 2, 0]
    
    grid = create_grid(h, w)
    flow12 = trans_coordinates - grid
    
    warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
    return warped_frame2, mask2, flow12

def compute_transformed_points(depth1: np.ndarray, transformation1: np.ndarray,
                                transformation2: np.ndarray, intrinsic1: np.ndarray,
                                intrinsic2: Optional[np.ndarray]):
    h, w = depth1.shape
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    transformation = np.matmul(transformation2, np.linalg.inv(transformation1))
    y1d = np.array(range(h))
    x1d = np.array(range(w))
    x2d, y2d = np.meshgrid(x1d, y1d)
    ones_2d = np.ones(shape=(h, w))
    ones_4d = ones_2d[:, :, None, None]
    pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]
    
    intrinsic1_inv = np.linalg.inv(intrinsic1)
    intrinsic1_inv_4d = intrinsic1_inv[None, None]
    intrinsic2_4d = intrinsic2[None, None]
    depth_4d = depth1[:, :, None, None]
    trans_4d = transformation[None, None]
    
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
    h, w, c = frame1.shape
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if flow12_mask is None:
        flow12_mask = np.ones(shape=(h, w), dtype=bool)
        
    grid = create_grid(h, w)
    trans_pos = flow12 + grid
    
    trans_pos_offset = trans_pos + 1.0
    trans_pos_floor = np.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
    
    trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_floor = np.clip(trans_pos_floor, a_min=0, a_max=[w + 1, h + 1])
    trans_pos_ceil = np.clip(trans_pos_ceil, a_min=0, a_max=[w + 1, h + 1])
    
    prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    
    sat_depth1 = np.clip(depth1, a_min=0, a_max=5000)
    log_depth1 = np.log(1 + sat_depth1)
    depth_weights = np.exp(log_depth1 / (log_depth1.max() + 1e-8) * 50)
    
    weight_nw = prox_weight_nw * mask1 * flow12_mask / (depth_weights + 1e-8)
    weight_sw = prox_weight_sw * mask1 * flow12_mask / (depth_weights + 1e-8)
    weight_ne = prox_weight_ne * mask1 * flow12_mask / (depth_weights + 1e-8)
    weight_se = prox_weight_se * mask1 * flow12_mask / (depth_weights + 1e-8)
    
    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]
    
    warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
    warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)
    
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)
    
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)
    
    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]
    
    mask = cropped_weights > 0.0
    mask2 = cropped_weights <= 0.6
    mask = mask * mask2
    
    with np.errstate(invalid='ignore'):
        warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / (cropped_weights[:, :, None] + 1e-8), 0)
        
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
    forward = normalize(target - camera_position)
    right = normalize(np.cross(up, forward))
    up = np.cross(forward, right)
    rotation = np.array([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1]
    ])
    translation = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])
    view_matrix = rotation.T @ translation
    return view_matrix

def generate_camera_poses_around_ellipse(num_poses, angle_step, major_radius, minor_radius, inverse=False):
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

def save_warped_image(image_path, depth_path, num_frames, radius, end_position):    
    MODEL_WIDTH = 1024
    MODEL_HEIGHT = 576
    
    original_image = PIL.Image.open(image_path)
    original_width, original_height = original_image.size
    
    image_o = original_image.resize((MODEL_WIDTH, MODEL_HEIGHT), PIL.Image.Resampling.LANCZOS)
    image = np.array(image_o)
    
    depth = np.load(depth_path).astype(np.float32)
    depth[depth < 1e-5] = 1e-5
    near, far = 0.0001, 500.
    depth = np.clip(depth, near, far)
    depth = cv2.resize(depth, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    focal = 320.
    K = np.eye(3)
    K[0,0] = focal
    K[1,1] = focal
    K[0,2] = MODEL_WIDTH / 2.0
    K[1,2] = MODEL_HEIGHT / 2.0
    
    start_position = [radius, 0, 0]
    poses, cams = generate_camera_poses(start_position, end_position, num_frames)
    
    pose_s = poses[0]
    cond_image = []
    masks = []
    i = 0
    for pose_t in poses[1:]:
        warped_frame2, mask2,flow12= forward_warp(image, None, depth, pose_s, pose_t, K, None)

        mask = 1-mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = np.repeat(mask[:,:,np.newaxis]*255.,repeats=3,axis=2)

        kernel = np.ones((5,5), np.uint8)
        mask_erosion = cv2.dilate(np.array(mask), kernel, iterations = 1)
        mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))

        mask_erosion_ = np.array(mask_erosion)/255.
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2))
        warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2*(1-mask_erosion_)))

        cond_image.append(warped_frame2)

        mask_erosion = np.mean(mask_erosion_,axis = -1)
        mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
        mask_erosion = np.mean(mask_erosion,axis=2)
        mask_erosion[mask_erosion < 0.2] = 0
        mask_erosion[mask_erosion >= 0.2] = 1
        masks.append(torch.from_numpy(mask_erosion).unsqueeze(0))

        i+=1
    masks = torch.cat(masks)
    return original_image, image_o, masks, cond_image, original_height, original_width, MODEL_HEIGHT, MODEL_WIDTH, K, poses

def svd_render(original_image, model_image, masks, cond_image, image_path, output_path, 
               lambda_ts, lr, weight_clamp, original_height, original_width, model_height, model_width,
               device: torch.device):  # 新增：指定device
    # 加载模型到指定GPU
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    pipe = pipe.to(device)  # 移到指定GPU
    
    # 调用管道时指定device
    frames = pipe(
        [model_image],
        temp_cond=cond_image,
        mask=masks,
        lambda_ts=lambda_ts,
        lr=lr,
        weight_clamp=weight_clamp,
        height=model_height,
        width=model_width,
        num_frames=25,
        num_inference_steps=100,
        decode_chunk_size=8,
        output_type="pil",
        device=device,  # 传入device
    ).frames[0]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    resized_frames = []
    for i, fr in enumerate(frames):
        resized_frame = fr.resize((original_width, original_height), PIL.Image.Resampling.LANCZOS)
        resized_frame.save(os.path.join(output_path, f"nvs_frame_{i:06d}.png"))
        resized_frames.append(resized_frame)
    
    # export_to_video(resized_frames, os.path.join(output_path, "nvs_generated.mp4"), fps=7)
    return resized_frames

def search_hypers(sigmas, save_path, is_main_process: bool = True):  # 新增：控制日志打印
    sigmas = sigmas[:-1]
    sigmas_max = max(sigmas)
    v2_list = np.arange(50, 1001, 50)
    v3_list = np.arange(10, 101, 10)
    v1_list = np.linspace(0.001, 0.009, 9)
    zero_count_default = 0
    index_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    v_optimized = None
    lambda_t_list_optimized = None

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
                            try:
                                lambda_t_1 = (-(2*v1 + k*Q) + ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                                lambda_t_2 = (-(2*v1 + k*Q) - ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                                v1_ = -v1
                                lambda_t_3 = (-(2*v1_ + k*Q) + ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                                lambda_t_4 = (-(2*v1_ + k*Q) - ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                                
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
                if flag:
                    zero_count = sum(1 for x in lambda_t_list if x > 0.5)
                    if zero_count > zero_count_default:
                        zero_count_default = zero_count
                        v_optimized = [v1, v2, v3]
                        lambda_t_list_optimized = lambda_t_list
    
    if v_optimized is not None and is_main_process:  # 仅主进程保存图片
        X = np.array(sigmas)
        Y = np.arange(0,25,1)
        X, Y = np.meshgrid(X, Y)
        lambda_t_list_optimized_np = np.array(lambda_t_list_optimized).reshape([len(sigmas),25])
        lambda_t_list_optimized = torch.tensor(lambda_t_list_optimized_np)
        Z = lambda_t_list_optimized
        z_upsampled = F.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='bilinear', align_corners=True)
        # save_path_img = os.path.join(save_path, f'lambad_{v_optimized[0]}_{v_optimized[1]}_{v_optimized[2]}.png')
        # image_numpy = z_upsampled[0].permute(1, 2, 0).numpy()
        # plt.figure()  
        # plt.imshow(image_numpy)  
        # plt.colorbar()  
        # plt.axis('off')  
        # plt.savefig(save_path_img, bbox_inches='tight', pad_inches=0.1)
        # plt.close() 
        return lambda_t_list_optimized
    else:
        if is_main_process:  # 仅主进程打印警告
            print("Warning: No valid lambda_t configuration found during hyperparameter search.")
        return torch.ones(len(sigmas), 25)

def bilinear_splatting(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                        flow12: Optional[np.ndarray], flow12_mask: Optional[np.ndarray], is_image: bool = False,
                        transformation1: Optional[np.ndarray] = None, transformation2: Optional[np.ndarray] = None,
                        intrinsic1: Optional[np.ndarray] = None, intrinsic2: Optional[np.ndarray] = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    h, w, c = frame1.shape
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if flow12_mask is None:
        flow12_mask = np.ones(shape=(h, w), dtype=bool)

    if transformation1 is not None and transformation2 is not None and intrinsic1 is not None:
        trans_points1, _ = compute_transformed_points(depth1, transformation1, transformation2, intrinsic1, intrinsic2)
        trans_coordinates = trans_points1[:, :, :2, 0] / (trans_points1[:, :, 2:3, 0] + 1e-8)
        grid = create_grid(h, w)
        flow12 = trans_coordinates - grid
        
    if flow12 is None:
        raise ValueError("Either 'flow12' or 'transformation' parameters must be provided.")

    grid = create_grid(h, w)
    trans_pos = flow12 + grid
    
    trans_pos_offset = trans_pos + 1.0
    trans_pos_floor = np.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
    
    trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_floor = np.clip(trans_pos_floor, a_min=0, a_max=[w + 1, h + 1])
    trans_pos_ceil = np.clip(trans_pos_ceil, a_min=0, a_max=[w + 1, h + 1])
    
    prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    
    sat_depth1 = np.clip(depth1, a_min=0, a_max=5000)
    log_depth1 = np.log(1 + sat_depth1)
    depth_weights = np.exp(log_depth1 / (log_depth1.max() + 1e-8) * 50)
    
    weight_nw = prox_weight_nw * mask1 * flow12_mask / (depth_weights + 1e-8)
    weight_sw = prox_weight_sw * mask1 * flow12_mask / (depth_weights + 1e-8)
    weight_ne = prox_weight_ne * mask1 * flow12_mask / (depth_weights + 1e-8)
    weight_se = prox_weight_se * mask1 * flow12_mask / (depth_weights + 1e-8)
    
    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]
    
    warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
    warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)
    
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)
    
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)
    
    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]
    
    mask = cropped_weights > 0.001
    
    with np.errstate(invalid='ignore', divide='ignore'):
        warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / (cropped_weights[:, :, None] + 1e-8), 0)
        
    if is_image:
        clipped_image = np.clip(warped_frame2, a_min=0, a_max=255)
        warped_frame2 = np.round(clipped_image).astype('uint8')
        
    return warped_frame2, mask

def save_specified_frames_and_depth(nvs_frames: List[PIL.Image.Image],
                                   image_path: str,
                                   depth_path: str,
                                   base_output_path: str,
                                   frame_pairs: List[Tuple[int, int]] = [(0,4), (0,6), (0,8), (0,10)]) -> None:
    print("--- 开始保存指定帧对和深度图 ---")
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    max_frame_idx = len(nvs_frames) - 1
    for ref_idx, target_idx in frame_pairs:
        if ref_idx > max_frame_idx or target_idx > max_frame_idx:
            raise ValueError(f"帧索引超出范围！最大帧索引为{max_frame_idx}，但请求了({ref_idx}, {target_idx})")
    
    for ref_idx, target_idx in frame_pairs:
        folder_suffix = f"{ref_idx:02d}{target_idx:02d}"
        folder_name = f"{image_name}_{folder_suffix}"
        folder_path = os.path.join(base_output_path, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        depth_folder_path = os.path.join(folder_path, "depth")
        os.makedirs(depth_folder_path, exist_ok=True)
        
        depth_filename = os.path.basename(depth_path)
        depth_save_path = os.path.join(depth_folder_path, depth_filename)
        
        shutil.copy2(depth_path, depth_save_path)
        print(f"  深度图已保存到: {depth_save_path}")
        
        ref_frame = nvs_frames[ref_idx]
        target_frame = nvs_frames[target_idx]
        
        ref_frame_save_path = os.path.join(folder_path, f"frame_{ref_idx:06d}.png")
        target_frame_save_path = os.path.join(folder_path, f"frame_{target_idx:06d}.png")
        
        ref_frame.save(ref_frame_save_path)
        target_frame.save(target_frame_save_path)
        
        print(f"  帧对 ({ref_idx}, {target_idx}) 已保存到: {folder_path}")
        print(f"    - 参考帧: {ref_frame_save_path}")
        print(f"    - 目标帧: {target_frame_save_path}")
    
    print("--- 指定帧对和深度图保存完成 ---")

def svd_render_and_postprocess(original_image, model_image, masks, cond_image, image_path, output_path, 
                               lambda_ts, lr, weight_clamp, original_height, original_width, model_height, model_width,
                               K, poses, depth_path=None, device: torch.device = None, args=None): 
    if args and args.is_main_process:
        print("--- 阶段 1: 开始 SVD 渲染 ---")
    nvs_output_path = os.path.join(output_path, "nvs_output")
    nvs_frames = svd_render(
        original_image, model_image, masks, cond_image, image_path, nvs_output_path, 
        lambda_ts, lr, weight_clamp, original_height, original_width, model_height, model_width,
        device=device  # 传入device
    )
    if args and args.is_main_process:
        print(f"--- SVD 渲染完成，结果保存在 {nvs_output_path} ---")

    if args and args.is_main_process:
        print("--- 阶段 2: 开始保存指定帧对和深度图 ---")
    frame_pairs = [(0, 4), (0, 6), (0, 8), (0, 10)]
    save_specified_frames_and_depth(
        nvs_frames=nvs_frames,
        image_path=image_path,
        depth_path=depth_path, 
        base_output_path=output_path,
        frame_pairs=frame_pairs
    )

    return nvs_frames

# ===================== 分布式主处理函数 =====================
def main_worker(proc_id, args):
    """每个GPU的工作函数"""
    # 把进程ID赋值给args，供init_distributed使用
    args.proc_id = proc_id
    # 初始化分布式环境（单卡会自动跳过）
    init_distributed(args)
    
    num_frames = 25
    img_exts = args.img_exts if isinstance(args.img_exts, list) else ['.png', '.jpg', '.jpeg']
    
    # ========== 收集所有样本（单卡/多卡分离逻辑） ==========
    sample_list = []
    # 只有多卡且为主进程，或单卡时，收集样本
    if args.is_main_process:
        input_folder = args.input_folder
        # 检查输入文件夹是否存在
        if not os.path.exists(input_folder):
            print(f"❌ 输入文件夹不存在: {input_folder}")
            return
        
        for img_file in os.listdir(input_folder):
            img_ext = os.path.splitext(img_file)[1].lower()
            if img_ext not in img_exts:
                continue
            image_path = os.path.join(input_folder, img_file)
            img_name = os.path.splitext(img_file)[0]
            depth_path = os.path.join(input_folder, 'depth', f"{img_name}.npy")
            if not os.path.exists(depth_path):
                print(f"⚠️  跳过 {img_file}：未找到对应的深度图 {depth_path}")
                continue
            sample_list.append((image_path, depth_path))
        
        total_samples = len(sample_list)
        print(f"\n📊 总样本数: {total_samples} | 卡数: {args.world_size} | 每卡平均样本数: {total_samples/args.world_size:.1f}")
        
        # 多卡场景：广播样本列表到所有进程；单卡场景直接使用
        if args.world_size > 1:
            sample_list_broadcast = [sample_list]
            dist.broadcast_object_list(sample_list_broadcast, src=0)
            sample_list = sample_list_broadcast[0]
    else:
        # 多卡非主进程：接收广播的样本列表（单卡不会走到这里）
        if args.world_size > 1:
            sample_list_broadcast = [[]]
            dist.broadcast_object_list(sample_list_broadcast, src=0)
            sample_list = sample_list_broadcast[0]
    
    # 无样本时直接退出
    if len(sample_list) == 0:
        print(f"⚠️  未找到有效样本（图片+深度图），退出处理")
        return
    
    # ========== 按rank划分样本 ==========
    rank = args.rank
    world_size = args.world_size
    samples_per_rank = len(sample_list) // world_size
    remainder = len(sample_list) % world_size
    start_idx = rank * samples_per_rank + min(rank, remainder)
    end_idx = start_idx + samples_per_rank + (1 if rank < remainder else 0)
    local_samples = sample_list[start_idx:end_idx]
    
    if args and args.is_main_process:
        print(f"\n🚀 各卡样本分配:")
        for r in range(world_size):
            s = r * samples_per_rank + min(r, remainder)
            e = s + samples_per_rank + (1 if r < remainder else 0)
            print(f"   GPU{r}: 样本数={e-s} (索引{s}-{e-1})")
    
    # ========== 处理当前卡的样本 ==========
    if args and args.is_main_process:
        print(f"\n💻 GPU{rank} 开始处理 {len(local_samples)} 个样本...")
    
    for idx, (image_path, depth_path) in enumerate(local_samples):
        img_file = os.path.basename(image_path)
        if args and args.is_main_process:
            print(f"\n=====================================")
            print(f"GPU{rank} 处理第 {idx+1}/{len(local_samples)} 个样本：{img_file}")
            print(f"=====================================")
        
        # 单样本输出路径
        img_name = os.path.splitext(img_file)[0]
        img_output_root = os.path.join(args.folder_path, img_name)
        main_output_path = os.path.join(img_output_root)
        os.makedirs(main_output_path, exist_ok=True)
        
        # 加载深度图
        try:
            original_depth = np.load(depth_path).astype(np.float32)
            original_depth[original_depth < 1e-5] = 1e-5
        except Exception as e:
            print(f"⚠️  GPU{rank} 跳过 {img_file}：深度图加载失败 - {e}")
            continue
        
        # 读取原始图像
        try:
            original_image_temp = PIL.Image.open(image_path)
            original_width_temp, original_height_temp = original_image_temp.size
        except Exception as e:
            print(f"⚠️  GPU{rank} 跳过 {img_file}：图片加载失败 - {e}")
            continue
        
        # 计算radius和end_position
        depth_resized = cv2.resize(original_depth, (original_width_temp, original_height_temp), interpolation=cv2.INTER_NEAREST)
        center_y, center_x = original_height_temp // 2, original_width_temp // 2
        Z_center = depth_resized[center_y, center_x]
        if args and args.is_main_process:
            print(f"📌 GPU{rank} {img_file} - 中心深度 Z_center: {Z_center:.4f} 米")
        
        radius = Z_center / 2
        end_position = (radius, 0, 0.6)
        if args and args.is_main_process:
            print(f"📌 GPU{rank} {img_file} - radius: {radius:.4f} 米 | end_position: {end_position}")
        
        # 生成warped图像
        try:
            original_image, model_image, masks, cond_image, original_height, original_width, model_height, model_width, K, poses = save_warped_image(
                image_path, 
                depth_path,
                num_frames, radius, end_position
            )
        except Exception as e:
            print(f"⚠️  GPU{rank} 跳过 {img_file}：生成warped图像失败 - {e}")
            continue
        
        if args and args.is_main_process:
            print(f"📌 GPU{rank} {img_file} - 原始分辨率: {original_width}x{original_height} | 模型分辨率: {model_width}x{model_height}")
        
        # 搜索lambda_ts（兜底处理）
        try:
            sigma_list = np.load('sigmas/sigmas_100.npy').tolist()
            lambda_ts = search_hypers(sigma_list, main_output_path, args.is_main_process)
        except FileNotFoundError:
            print(f"⚠️  GPU{rank} {img_file} - Sigma文件未找到，使用默认lambda_ts")
            lambda_ts = torch.ones(100, 25).to(args.device)
        except Exception as e:
            print(f"⚠️  GPU{rank} {img_file} - 搜索lambda_ts失败，使用默认值: {e}")
            lambda_ts = torch.ones(100, 25).to(args.device)
        
        # 执行SVD渲染和后处理
        try:
            nvs_frames = svd_render_and_postprocess(
                original_image, model_image, masks, cond_image, image_path, main_output_path,
                lambda_ts, args.lr, args.weight_clamp, original_height, original_width, model_height, model_width,
                K, poses,
                depth_path=depth_path,
                device=args.device,
                args=args
            )
            if args and args.is_main_process:
                print(f"✅ GPU{rank} {img_file} 处理完成！结果保存到: {main_output_path}")
                # 打印帧对文件夹路径
                for suffix in ["0004", "0006", "0008", "0010"]:
                    frame_pair_folder = os.path.join(main_output_path, f"{img_name}_{suffix}")
                    print(f"   - 帧对文件夹: {frame_pair_folder}")
        except Exception as e:
            print(f"❌ GPU{rank} {img_file} SVD渲染失败 - {e}")
            traceback.print_exc()
            continue
    
    # ========== 处理完成（单卡/多卡分离清理逻辑） ==========
    if args.world_size > 1:
        # 多卡：同步所有进程后销毁分布式环境
        dist.barrier()
        dist.destroy_process_group()
    
    if args and args.is_main_process:
        print(f"\n🎉 GPU{rank} 本地样本处理完成！共处理 {len(local_samples)} 个样本")
        print(f"📁 总输出根路径：{args.folder_path}")


# ===================== 主函数（入口） =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=10.0, help='Learning rate for optimization.')
    parser.add_argument('--weight_clamp', type=float, default=0.4, help='Weight clamp value.')
    parser.add_argument('--input_folder', type=str, required=True, help='批量处理的根文件夹（包含原始图片和depth子文件夹）')
    parser.add_argument('--folder_path', type=str, required=True, help='Base folder path containing depth maps and for outputs.')
    parser.add_argument('--iteration', type=str, default='000', help='Iteration name for output subfolders.')
    parser.add_argument('--img_exts', nargs='+', default=['.png', '.jpg', '.jpeg'], help='需要处理的图片后缀（空格分隔）')
    # 分布式相关参数（由torchrun自动传入，无需手动指定）
    
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count()
    print(f"📌 检测到 {num_gpus} 个GPU，开始启动多进程...")
    mp.spawn(
        main_worker, 
        args=(args,),  # 注意逗号，确保是元组
        nprocs=num_gpus,  # 启动的进程数（等于GPU数）
        join=True
    )