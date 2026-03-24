import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, DDIMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from einops import rearrange, repeat
from autoencode_kl import AutoencoderKL

# from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler

def make_1step_sched(pretrained_name=None):
    """
    构建“单步”DDPM调度器。

    这里并不是完整多步采样，而是把 scheduler 的推理步数设为 1，
    让 UNet 只执行一次去噪更新（与 SD-Turbo / Difix 的快速路径一致）。
    """
    if pretrained_name is None:
        pretrained_name = "stabilityai/sd-turbo"
    
    noise_scheduler_1step = DDPMScheduler.from_pretrained(pretrained_name, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    """
    重载 VAE encoder.forward：
    - 保留每个 down block 的中间特征到 self.current_down_blocks；
    - 供 decoder 阶段做 skip connection 注入。
    """
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    """
    重载 VAE decoder.forward：
    - 用 encoder 存下来的特征（incoming_skip_acts）经 skip_conv_* 映射后注入 up block；
    - 相当于给 SD 原生 VAE decoder 增加“可学习跳连”，提升复原细节能力。
    """
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    
    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)
        
    optimizer.load_state_dict(sd["optimizer"])
    
    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_difix.vae.state_dict().items() if "lora" in k or "skip" in k}
    
    sd["optimizer"] = optimizer.state_dict()   
    
    torch.save(sd, outf)


class DifixRefWithPose(torch.nn.Module):
    """
    DifixRefWithPose: 文本条件 + 图像条件 + 可选相对位姿条件 的单步扩散复原模型。

    关键路径：
    1) 文本 prompt -> CLIP text embedding
    2) 图像 x -> VAE latent z
    3) 可选 relative_pose -> Positional Encoding -> pose_mlp -> pose token
    4) 将 pose token 作为“额外 prompt token”拼接到文本 token 序列后
    5) UNet 单步去噪 + scheduler.step
    6) VAE decode（带 skip）得到输出图像
    """
    def __init__(self, pretrained_name=None,                  
                 pretrained_path=None, 
                 ckpt_folder="checkpoints", 
                 lora_rank_vae=4, 
                 mv_unet=False, 
                 timestep=999,
                 deterministic_vae_encode: bool = False,
                 deterministic_scheduler_step: bool = False):
        super().__init__()
        
        if pretrained_name=="None":
            pretrained_name = None
        
        if pretrained_name is None:
            pretrained_name = "stabilityai/sd-turbo"

        if pretrained_path =="None":
            pretrained_path = None
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, 
                                                       subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_name, 
                                                          subfolder="text_encoder").cuda()
        
        self.sched = make_1step_sched(pretrained_name)
        

        vae = AutoencoderKL.from_pretrained(pretrained_name, subfolder="vae", trust_remote_code=True)
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        
        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            # vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
        
        else:
            target_modules_vae = []
            
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                            "to_k", "to_q", "to_v", "to_out.0",
                        ]
                        
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
                
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            
            
        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.deterministic_vae_encode = deterministic_vae_encode
        self.deterministic_scheduler_step = deterministic_scheduler_step
        # -------- Pose prompt 分支 --------
        # pose_num_freqs: 对 4x4 pose 展平后的每个标量做多频率正余弦编码的频率数量。
        # 编码维度 = 原始16维 + sin/cos双通道 * 16维 * 频率数
        #         = 16 * (2 * pose_num_freqs + 1)
        self.pose_num_freqs = 8
        self.pose_embed_dim = self.text_encoder.config.hidden_size
        pose_in_dim = 16 * (2 * self.pose_num_freqs + 1)
        # pose_mlp 输出维度与文本 token 维度一致，便于直接拼接到 encoder_hidden_states。
        self.pose_mlp = torch.nn.Sequential(
            torch.nn.Linear(pose_in_dim, self.pose_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.pose_embed_dim, self.pose_embed_dim),
        )
        self.pose_mlp.to("cuda")

        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        if deterministic_vae_encode or deterministic_scheduler_step:
            print(
                f"DifixRef deterministic forward: vae_encode={deterministic_vae_encode}, "
                f"ddpm_step_mean_only={deterministic_scheduler_step}"
            )
        print("="*50)
        


    def _ddpm_prev_sample_mean_only(self, model_output, sample):
        """
        DDPM 反向一步“确定性”版本：
        - 复用 diffusers DDPMScheduler.step 的前 1~5 步；
        - 跳过第6步随机方差噪声注入（randn_tensor）；
        - 返回纯均值轨迹下的 prev_sample。

        作用：减少扩散随机性，提升像素回归任务（PSNR/SSIM）的稳定性。
        """
        sched = self.sched
        ts = self.timesteps[0]
        t = int(ts.item()) if torch.is_tensor(ts) else int(ts)
        prev_t = sched.previous_timestep(t)
        prev_t_idx = int(prev_t.item()) if torch.is_tensor(prev_t) else int(prev_t)

        if model_output.shape[1] == sample.shape[1] * 2 and sched.config.variance_type in ["learned", "learned_range"]:
            model_output, _predicted_variance = torch.split(model_output, sample.shape[1], dim=1)

        acp = sched.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        one = sched.one.to(device=sample.device, dtype=sample.dtype)

        alpha_prod_t = acp[t]
        alpha_prod_t_prev = acp[prev_t_idx] if prev_t_idx >= 0 else one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if sched.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif sched.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif sched.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type {sched.config.prediction_type} not supported for deterministic DDPM step."
            )

        if sched.config.thresholding:
            pred_original_sample = sched._threshold_sample(pred_original_sample)
        elif sched.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -sched.config.clip_sample_range, sched.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        return pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.pose_mlp.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.pose_mlp.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.pose_mlp.train()
        self.unet.requires_grad_(True)
        self.pose_mlp.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def _pose_positional_encoding(self, relative_pose: torch.Tensor) -> torch.Tensor:
        """
        将相对位姿做 Fourier-style positional encoding。

        输入:
            relative_pose: [B, V, 4, 4]
        过程:
            1) 展平为 [B, V, 16]
            2) 拼接 [x, sin(2^k x), cos(2^k x)] (k=0..pose_num_freqs-1)
        输出:
            [B, V, 16 * (2*F + 1)]
        """
        # relative_pose: [B, V, 4, 4] -> flatten to [B, V, 16]
        pose_flat = relative_pose.float().reshape(relative_pose.shape[0], relative_pose.shape[1], -1)
        enc = [pose_flat]
        for i in range(self.pose_num_freqs):
            freq = 2.0 ** i
            angle = pose_flat * freq
            enc.append(torch.sin(angle))
            enc.append(torch.cos(angle))
        return torch.cat(enc, dim=-1)

    def _pose_to_prompt_tokens(self, relative_pose: torch.Tensor) -> torch.Tensor:
        """
        位姿 -> prompt token。

        输入:
            [B, V, 4, 4]
        输出:
            [B*V, 1, D]
        说明:
            - 每个视图生成1个pose token；
            - D 与 CLIP token hidden size 对齐；
            - 后续会与文本 token 在 sequence 维拼接。
        """
        # [B, V, 4, 4] -> [B*V, 1, D]
        pose_pe = self._pose_positional_encoding(relative_pose)
        pose_embed = self.pose_mlp(pose_pe)
        pose_embed = rearrange(pose_embed, "b v d -> (b v) 1 d")
        return pose_embed



    def forward(self, x, 
                relative_pose=None,
                timesteps=None, 
                prompt=None, 
                prompt_tokens=None):
        """
        前向路径（训练与推理共享）。

        Args:
            x: [B, V, C, H, W]，其中 V 通常是主视角+参考视角。
            relative_pose: [B, V, 4, 4] 或 None。
                - 若提供：会生成 pose token 并拼到文本 token 序列后；
                - 若不提供：退化为普通 Difix 文本条件。
            timesteps: 目前内部固定 self.timesteps（单步），此参数保持接口兼容。
            prompt / prompt_tokens: 二选一。
        Returns:
            output_image: [B, V, C, H, W]，范围约 [-1, 1]。
        """
        
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        
        # 1) 文本编码：得到 [B, N_text, D]
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
                                
        # 2) 图像编码到 latent：先展平视图维，统一按 (B*V) 批处理
        num_views = x.shape[1]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        latent_dist = self.vae.encode(x).latent_dist
        if self.deterministic_vae_encode:
            # 确定性路径：使用后验均值，避免 sample() 的随机波动
            z = latent_dist.mode() * self.vae.config.scaling_factor
        else:
            z = latent_dist.sample() * self.vae.config.scaling_factor

        # 3) 文本 token 在视图维复制，与 (B*V) 对齐
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        if relative_pose is not None:
            # 4) relative_pose -> pose token，并在 token 序列维拼接
            #    拼接后 encoder_hidden_states 形状: [B*V, N_text + 1, D]
            pose_tokens = self._pose_to_prompt_tokens(relative_pose.to(caption_enc.device, dtype=caption_enc.dtype))
            caption_enc = torch.cat([caption_enc, pose_tokens], dim=1)
        
        # 5) 单步扩散：UNet预测 + scheduler更新
        unet_input = z
        # Scale model input according to the current timestep
        unet_input = self.sched.scale_model_input(unet_input, self.timesteps[0])
        
        model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
        if self.deterministic_scheduler_step:
            # 确定性路径：不加随机方差噪声
            z_denoised = self._ddpm_prev_sample_mean_only(model_pred, z)
        else:
            z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample

        # 6) 带 skip 的 VAE decode 回像素空间，并恢复 [B, V, C, H, W]
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        
        return output_image
    
    
    def _prepare_sample_input(self, image, ref_image, width, height):
        """
        统一处理 sample 输入图像，返回模型所需的 x 以及原图尺寸。

        Returns:
            x: [1, V, 3, H, W]，V=1(无ref) 或 V=2(有ref)
            input_width, input_height: 用于把输出 resize 回原图尺寸
            has_ref: bool，是否包含参考图
        """
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8

        image = image.resize((new_width, new_height), Image.LANCZOS)

        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
            has_ref = False
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()
            has_ref = True

        return x, input_width, input_height, has_ref

    def _default_relative_pose_for_views(self, expected_views: int, device) -> torch.Tensor:
        """
        当调用方未提供 relative_pose 时，构造默认位姿：
        - V=1: [I]
        - V=2: [I, I]
        """
        eye = torch.eye(4, dtype=torch.float32, device=device)
        return eye.unsqueeze(0).repeat(expected_views, 1, 1).unsqueeze(0)

    def sample(
        self,
        image,
        width,
        height,
        ref_image=None,
        timesteps=None,
        prompt=None,
        prompt_tokens=None,
        relative_pose=None,
    ):
        """
        默认推理入口（优先走 pose 条件分支）：
        - 若提供 relative_pose：直接使用；
        - 若不提供：自动构造单位矩阵位姿并调用 sample_with_pose。
        """
        x, _input_width, _input_height, has_ref = self._prepare_sample_input(
            image=image, ref_image=ref_image, width=width, height=height
        )
        expected_views = 2 if has_ref else 1
        if relative_pose is None:
            relative_pose = self._default_relative_pose_for_views(expected_views, x.device)

        return self.sample_with_pose(
            image=image,
            width=width,
            height=height,
            relative_pose=relative_pose,
            ref_image=ref_image,
            timesteps=timesteps,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
        )

    def sample_with_pose(
        self,
        image,
        width,
        height,
        relative_pose=None,
        ref_image=None,
        timesteps=None,
        prompt=None,
        prompt_tokens=None,
    ):
        """
        带位姿条件的推理接口（成套 sample）。

        Args:
            image: PIL.Image，主输入图
            width, height: 网络输入尺寸
            relative_pose: 支持三种输入形状（默认 None 时自动使用单位矩阵位姿）
                - [4,4]         : 无ref单视图，自动扩成 [1,1,4,4]
                - [V,4,4]       : 单batch多视图，自动扩成 [1,V,4,4]
                - [1,V,4,4]     : 直接使用
              其中 V 必须与 x 的视图数一致（无ref为1，有ref为2）
            ref_image: PIL.Image 或 None
            timesteps/prompt/prompt_tokens: 与 forward 一致
        Returns:
            output_pil: 主视图输出（第0视图）
        """
        x, input_width, input_height, has_ref = self._prepare_sample_input(
            image=image, ref_image=ref_image, width=width, height=height
        )
        expected_views = 2 if has_ref else 1

        if relative_pose is None:
            relative_pose = self._default_relative_pose_for_views(expected_views, x.device)
        if not torch.is_tensor(relative_pose):
            relative_pose = torch.as_tensor(relative_pose, dtype=torch.float32)
        relative_pose = relative_pose.to(device=x.device, dtype=torch.float32)

        if relative_pose.dim() == 2:
            relative_pose = relative_pose.unsqueeze(0).unsqueeze(0)
        elif relative_pose.dim() == 3:
            relative_pose = relative_pose.unsqueeze(0)
        elif relative_pose.dim() != 4:
            raise ValueError(
                f"relative_pose must be [4,4], [V,4,4], or [1,V,4,4], got {tuple(relative_pose.shape)}"
            )

        if relative_pose.shape[-2:] != (4, 4):
            raise ValueError(f"relative_pose last dims must be (4,4), got {tuple(relative_pose.shape)}")
        if relative_pose.shape[0] != 1:
            raise ValueError(f"sample_with_pose expects batch=1 pose, got shape {tuple(relative_pose.shape)}")
        if relative_pose.shape[1] != expected_views:
            raise ValueError(
                f"relative_pose views ({relative_pose.shape[1]}) != expected views ({expected_views})"
            )

        output_image = self.forward(
            x,
            relative_pose=relative_pose,
            timesteps=timesteps,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
        )[:, 0]

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        return output_pil




    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        
        sd["optimizer"] = optimizer.state_dict()
        
        torch.save(sd, outf)


