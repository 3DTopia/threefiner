from diffusers import (
    PNDMScheduler,
    DDIMScheduler,
    IFPipeline,
    IFSuperResolutionPipeline,
)



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IF2(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        # model_key = "DeepFloyd/IF-II-L-v1.0",
        model_key = "DeepFloyd/IF-II-M-v1.0",
        t_range=[0.02, 0.50],
    ):
        super().__init__()

        self.device = device
        self.model_key = model_key
        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = IFSuperResolutionPipeline.from_pretrained(
            model_key, variant="fp16", torch_dtype=torch.float16, 
            watermarker=None, safety_checker=None, requires_safety_checker=False,
        )

        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        self.scheduler = pipe.scheduler
        self.image_noising_scheduler = pipe.image_noising_scheduler

        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
        
    
    def encode_text(self, prompt):
        # prompt: [str]
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
        self,
        pred_rgb,
        ori_rgb,
        step_ratio=None,
        guidance_scale=50,
        vers=None, hors=None,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        ori_rgb = ori_rgb.to(self.dtype)

        images = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False) * 2 - 1

        with torch.no_grad():
            max_t = torch.full((batch_size,), self.max_step, dtype=torch.long, device=self.device)

            # images_upscaled = images.clone()
            images_upscaled = F.interpolate(ori_rgb, (256, 256), mode="bilinear", align_corners=False).clamp(0, 1) * 2 - 1
            noise = torch.randn_like(images_upscaled)
            images_upscaled = self.image_noising_scheduler.add_noise(images_upscaled, noise, max_t)
            
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1).to(self.dtype)

            ######### debug
            # imagesx = self.produce_imgs(
            #     images_upscaled=images_upscaled,
            #     max_t=max_t,
            #     images=torch.randn_like(images),
            #     num_inference_steps=50,
            #     guidance_scale=4.0,
            # )  # [1, 3, 64, 64]
            # import kiui
            # kiui.vis.plot_image(images_upscaled * 0.5 + 0.5)
            # kiui.vis.plot_image(imagesx * 0.5 + 0.5)
            #########
            
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)
            # pred noise
            model_input = torch.cat([images_noisy, images_upscaled], dim=1)
            model_input = torch.cat([model_input] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * 2)
            max_tt = torch.cat([max_t] * 2)

            if hors is None:
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'

                embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])

            noise_pred = self.unet(
                model_input, tt, encoder_hidden_states=embeddings, class_labels=max_tt,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
            noise_pred_cond, predicted_variance = noise_pred_cond.split(model_input.shape[1] // 2, dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
            # grad = grad_norm.clamp(max=0.1) * grad / grad_norm
        
            target = (images - grad).detach()

        loss = 0.5 * F.mse_loss(images, target, reduction='sum') / images.shape[0]

        return loss

    @torch.no_grad()
    def produce_imgs(
        self,
        images_upscaled,
        max_t,
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=4.0,
        images=None,
    ):
        if images is None:
            images = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height,
                    width,
                ),
                device=self.device,
            )
        
        batch_size = images.shape[0]

        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the images if we are doing classifier-free guidance to avoid doing two forward passes.
            model_input = torch.cat([images, images_upscaled], dim=1)
            model_input = torch.cat([model_input] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            max_tt = torch.cat([max_t] * 2)
            
            # predict the noise residual
            noise_pred = self.unet(
                model_input, t, encoder_hidden_states=embeddings, class_labels=max_tt,
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
            noise_pred_cond, predicted_variance = noise_pred_cond.split(model_input.shape[1] // 2, dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        return images

    
    def prompt_to_img(
        self,
        images_upscaled,
        max_t,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=4.0,
        images=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img images
        images = self.produce_imgs(
            images_upscaled=images_upscaled,
            max_t=max_t,
            height=height,
            width=width,
            images=images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        
        # Img to Numpy
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        return images


if __name__ == "__main__":
    import kiui
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument("--vram_O", action="store_true", help="optimization for low VRAM usage")
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    kiui.seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = IF2(device, opt.fp16, opt.vram_O)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()