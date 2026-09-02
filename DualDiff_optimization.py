# =============================================================================
# Import required libraries
# =============================================================================
import os
import numpy as np
import cv2
from PIL import Image
import gc   

import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


import lpips

from criteria.cosine_loss import AdaptiveCosineLoss
from criteria.cosine_loss import CosineLoss
from criteria.nce_loss import NCELoss
from attention_control_idea import AttentionControlEdit
from utils import *


@torch.enable_grad()
class Adversarial_Opt:
    def __init__(self, args, model):
        self.device = args.device
        self.dataloader = args.dataloader
        
        self.diff_model = model
        
        

        self.diff_model.vae.requires_grad_(False)
        self.diff_model.text_encoder.requires_grad_(False)
        self.diff_model.unet.requires_grad_(False)
        

        self.diff_model.unet.train() 


        self.source_dir = args.source_dir
        self.protected_image_dir = args.protected_image_dir
        self.comparison_null_text = args.comparison_null_text
        self.target_choice = args.target_choice
        self.is_makeup = args.is_makeup
        self.source_text = args.source_text
        self.makeup_prompt = args.makeup_prompt
        self.MTCNN_cropping = args.MTCNN_cropping
        self.is_obfuscation = args.is_obfuscation
        self.image_size = args.image_size
        
        self.prot_steps = args.prot_steps
        self.diffusion_steps = args.diffusion_steps
        self.start_step = args.start_step
        

        self.adv_optim_weight = args.adv_optim_weight
        self.makeup_weight = args.makeup_weight
        
        self.semantic_weight = args.semantic_weight
        self.lpips_weight = args.lpips_weight
        self.reg_weight = args.reg_weight
        self.lr_latent = args.lr_latent
        self.lr_emb = args.lr_emb


        print("Loading Wavelet Loss model...")
        self.wavelet_loss_fn = WaveletHaarLoss(device=self.device)
        self.wavelet_weight = getattr(args, 'wavelet_weight', 10.0)
        self.wavelet_weight = args.wavelet_weight
        self.mid_block_hook_handle = None

        print("Loading LPIPS model...")
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.device).eval()
        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False
            
        self.augment = transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5)
        
        self.cosine_loss = AdaptiveCosineLoss(self.is_obfuscation, temperature=0.1).to(self.device)
        # Loss Functions
        #self.cosine_loss = CosineLoss(self.is_obfuscation)
        self.nce_loss = NCELoss(self.device, clip_model="ViT-B/32")
        
        # FR Models
        self.surrogate_models = load_FR_models(args, args.surrogate_model_names)
        self.test_model_name = args.test_model_name

    def get_FR_embeddings(self, image):
        features = []
        for model_name in self.surrogate_models.keys():
            input_size = self.surrogate_models[model_name][0]
            fr_model = self.surrogate_models[model_name][1]
            emb_source = fr_model(F.interpolate(image, size=input_size, mode='bilinear'))
            features.append(emb_source)
        return features

    def set_attention_control(self, controller):
        def ca_forward(self, place_in_unet):
            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = sim.softmax(dim=-1)
                
                attn = controller(attn, is_cross, place_in_unet)

                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)
                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out
            return forward

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
        controller.num_att_layers = cross_att_count

        def get_middle_feature_hook(controller):
            def hook(module, input, output):
                controller.middle_feature = output
            return hook
        
        self.mid_block_hook_handle = self.diff_model.unet.mid_block.register_forward_hook(
            get_middle_feature_hook(controller)
        )

    def reset_attention_control(self):
        def ca_forward(self):
            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = sim.softmax(dim=-1)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)
                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out
            return forward

        def register_recr(net_):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_)
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    register_recr(net__)   

        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                register_recr(net[1])
            elif "up" in net[0]:
                register_recr(net[1])
            elif "mid" in net[0]:
                register_recr(net[1])

    def diffusion_step(self, latent, context, t, explicit_batch=False):
        if not explicit_batch:
            latent_input = torch.cat([latent] * 2)
            noise_pred = self.diff_model.unet(
                latent_input, t, encoder_hidden_states=context)["sample"]
            noise_pred, _ = noise_pred.chunk(2)
        else:
            noise_pred = self.diff_model.unet(
                latent, t, encoder_hidden_states=context)["sample"]
            
        return self.diff_model.scheduler.step(noise_pred, t, latent)["prev_sample"]

    def null_text_embeddings(self):
        uncond_input = self.diff_model.tokenizer([""],
                                                 padding="max_length",
                                                 max_length=self.diff_model.tokenizer.model_max_length,
                                                 return_tensors="pt")
        return self.diff_model.text_encoder(uncond_input.input_ids.to(self.device))[0]

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            generator = torch.Generator().manual_seed(8888)
            gpu_generator = torch.Generator(device=image.device)
            gpu_generator.manual_seed(generator.initial_seed())
            latents = self.diff_model.vae.encode(image).latent_dist.sample(generator=gpu_generator)
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latent):
        latent = 1 / 0.18215 * latent
        image = self.diff_model.vae.decode(latent)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def ddim_inversion(self, image):
        uncond_embeddings = self.null_text_embeddings()
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)
        latent = self.image2latent(image)
        all_latents = [latent]
        
        torch.cuda.empty_cache()
        
        for i in tqdm(range(self.diffusion_steps - 1), desc="Inverting"):
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - i - 1]
            noise_pred = self.diff_model.unet(latent, t, encoder_hidden_states=uncond_embeddings)["sample"]
            
            next_timestep = t + self.diff_model.scheduler.config.num_train_timesteps // self.diff_model.scheduler.num_inference_steps
            alpha_bar_next = self.diff_model.scheduler.alphas_cumprod[next_timestep] \
                if next_timestep <= self.diff_model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
            
            reverse_x0 = (1 / torch.sqrt(self.diff_model.scheduler.alphas_cumprod[t]) * (
                latent - noise_pred * torch.sqrt(1 - self.diff_model.scheduler.alphas_cumprod[t])))
            
            latent = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred
            all_latents.append(latent)

        return all_latents

    def visualize(self, image_name, real_image, latents, controller, mask_pixel=None):
        if latents.shape[0] == 2:
            adv_latent = latents[1:2]
        else:
            adv_latent = latents

        adversarial_image = self.latent2image(adv_latent)
        

        if mask_pixel is not None:

            mask_np = mask_pixel.detach().cpu().permute(0, 2, 3, 1).numpy() # (1, 256, 256, 1)
            

            real_img_np = real_image.detach().cpu().permute(0, 2, 3, 1).numpy() # (1, 256, 256, 3) float [0, 1]
            if real_img_np.min() < 0:
                real_img_np = (real_img_np / 2 + 0.5) 
            real_img_np = (real_img_np * 255).astype(np.uint8)

            blended = adversarial_image.astype(np.float32) * mask_np + \
                      real_img_np.astype(np.float32) * (1.0 - mask_np)
            
            adversarial_image = blended.clip(0, 255).astype(np.uint8)
        # ==========================


        # result_dir = self.protected_image_dir + '/' + \
        #     self.test_model_name[0] + '/' + \
        #     self.target_choice + '/' + image_name
        result_dir = self.protected_image_dir + '/' + \
        self.test_model_name[0] + '/' + self.target_choice + '/' + f"step_{self.start_step}_iter{self.prot_steps}_lr_{self.lr_emb}_adv_{self.adv_optim_weight}_wav_{self.wavelet_weight}_reg_{self.reg_weight}_lpips_{self.lpips_weight}" + '/' + image_name
        adversarial_img = cv2.cvtColor(adversarial_image[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_dir + ".png", adversarial_img)


    def attacker(self,
                 image,
                 image_name,
                 source_embeddings,
                 target_embeddings,
                 controller,
                 null_text_dir=None,
                 bb_src1=None):
        
        inversion_latents = self.ddim_inversion(image)
        inversion_latents = inversion_latents[::-1] # T -> 0
        
        start_latent = inversion_latents[self.start_step - 1]


        latent = start_latent.clone().detach().to(torch.float32)
        latent.requires_grad_(True)

        with torch.no_grad():
            base_null_emb = self.null_text_embeddings().detach().to(torch.float32) 


        mask_pixel = torch.zeros_like(image)
        mask_latent = torch.zeros_like(start_latent)
        
        if bb_src1 is not None and self.MTCNN_cropping:
            h_s, h_e = round(bb_src1[1]), round(bb_src1[3])
            w_s, w_e = round(bb_src1[0]), round(bb_src1[2])
            

            pad_pixel = 5
            p_h_s, p_h_e = h_s + pad_pixel, h_e - pad_pixel
            p_w_s, p_w_e = w_s + pad_pixel, w_e - pad_pixel
            
            p_h_s, p_w_s = max(0, p_h_s), max(0, p_w_s)
            p_h_e, p_w_e = min(image.shape[2], p_h_e), min(image.shape[3], p_w_e)

            if p_h_e > p_h_s and p_w_e > p_w_s:
                mask_pixel[:, :, p_h_s:p_h_e, p_w_s:p_w_e] = 1.0
            
            scale = 64 / self.image_size
            l_h_s, l_h_e = int(h_s * scale), int(h_e * scale)
            l_w_s, l_w_e = int(w_s * scale), int(w_e * scale)
            
            pad_latent = 1
            l_h_s, l_h_e = l_h_s + pad_latent, l_h_e - pad_latent
            l_w_s, l_w_e = l_w_s + pad_latent, l_w_e - pad_latent
            
            l_h_s, l_w_s = max(0, l_h_s), max(0, l_w_s)
            l_h_e, l_w_e = min(64, l_h_e), min(64, l_w_e)
            
            if l_h_e > l_h_s and l_w_e > l_w_s:
                mask_latent[:, :, l_h_s:l_h_e, l_w_s:l_w_e] = 1.0

            pixel_blur = transforms.GaussianBlur(kernel_size=51, sigma=10.0)
            mask_pixel = pixel_blur(mask_pixel)
            
            latent_blur = transforms.GaussianBlur(kernel_size=7, sigma=1.5)
            mask_latent = latent_blur(mask_latent)
            
            if mask_pixel.max() > 0:
                mask_pixel = mask_pixel / mask_pixel.max()
            if mask_latent.max() > 0:
                mask_latent = mask_latent / mask_latent.max()

        else:
            print("Warning: No face detected for masking. Using full image.")
            mask_pixel[:] = 1.0
            mask_latent[:] = 1.0

        mask_pixel = mask_pixel.to(self.device)
        mask_latent = mask_latent.to(self.device)
        

        hook_handle = latent.register_hook(lambda grad: grad * mask_latent)


        optimizer = optim.AdamW([
            {'params': [latent], 'lr': self.lr_latent}
        ])

        self.set_attention_control(controller)
        init_latent = start_latent.clone().detach().to(torch.float32)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Starting Single-Variable Optimized Attack (Steps: {self.prot_steps})...")


        for step_idx in tqdm(range(self.prot_steps), desc="Optimizing"):
            controller.loss = 0
            controller.reset()
            controller.middle_feature = None 

            latents_input = torch.cat([init_latent, latent]) 


            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                
               
                emb_batch = torch.cat([base_null_emb, base_null_emb])
                
                model_dtype = self.diff_model.unet.dtype
                latents_input_model = latents_input.to(model_dtype)
                emb_batch_model = emb_batch.to(model_dtype)
                
                latents_input = self.diffusion_step(latents_input_model, emb_batch_model, t, explicit_batch=True)
                latents_input = latents_input.to(torch.float32)

 
            latents_decode = latents_input.to(self.diff_model.vae.dtype)
            out_image = self.diff_model.vae.decode(1 / 0.18215 * latents_decode)['sample']
            img_adv = out_image[1:2] 
            
 
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # 1. Adv Loss
            img_adv_for_fr = img_adv
            if self.MTCNN_cropping and bb_src1 is not None:
                h_s, h_e = round(bb_src1[1]), round(bb_src1[3])
                w_s, w_e = round(bb_src1[0]), round(bb_src1[2])

                h_s = max(0, h_s)
                h_e = min(img_adv.shape[2], h_e)
                w_s = max(0, w_s)
                w_e = min(img_adv.shape[3], w_e)
                
                if (h_e - h_s) > 0 and (w_e - w_s) > 0:
                    img_adv_for_fr = img_adv[:, :, h_s:h_e, w_s:w_e]

            diversity_counts = 5 
            total_adv_loss = 0.0
            
            for _ in range(diversity_counts):
                img_aug = input_diversity(img_adv_for_fr, resize_rate=0.9, diversity_prob=0.9)
                if torch.rand(1).item() < 0.5:
                    img_aug = torch.flip(img_aug, dims=[3])
                
                output_embeddings = self.get_FR_embeddings(img_aug)
                current_loss = self.cosine_loss(output_embeddings, target_embeddings, source_embeddings)
                total_adv_loss += current_loss
            
            adv_loss = (total_adv_loss / diversity_counts) * self.adv_optim_weight
            loss = loss + adv_loss
            
            # 2. Structure Loss (SA)
            self_attn_loss = controller.loss 
            loss = loss + self_attn_loss
            
            # 3. LPIPS Loss
            if img_adv.shape[-2:] != image.shape[-2:]:
               lpips_target = F.interpolate(image, size=img_adv.shape[-2:], mode='bilinear')
            else:
               lpips_target = image
            
            lpips_val = self.lpips_loss_fn(img_adv * mask_pixel, lpips_target * mask_pixel).mean() * self.lpips_weight
            loss = loss + lpips_val
            
            # 4. Wavelet Loss
            if img_adv.shape[-2:] != image.shape[-2:]:
               wav_target = F.interpolate(image, size=img_adv.shape[-2:], mode='bilinear')
            else:
               wav_target = image

            wavelet_loss_val = self.wavelet_loss_fn(img_adv * mask_pixel, wav_target * mask_pixel) * self.wavelet_weight
            loss = loss + wavelet_loss_val




            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            if step_idx % 5 == 0:
                print(f"Step {step_idx}: Adv: {adv_loss.item():.4f}, SA:{self_attn_loss.item():.4f}, Wav:{wavelet_loss_val.item():.4f}, LPIPS:{lpips_val.item():.4f}, Total: {loss.item():.4f}")
            
            controller.middle_feature = None


        print("Generating final result...")
        with torch.no_grad():
            controller.loss = 0
            controller.reset()
            latents_final = torch.cat([init_latent, latent])
            
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                
                emb_batch = torch.cat([base_null_emb, base_null_emb])
                
                model_dtype = self.diff_model.unet.dtype
                latents_final = latents_final.to(model_dtype)
                emb_batch = emb_batch.to(model_dtype)
                
                latents_final = self.diffusion_step(latents_final, emb_batch, t, explicit_batch=True)

        self.reset_attention_control()
        if self.mid_block_hook_handle is not None:
            self.mid_block_hook_handle.remove()
            self.mid_block_hook_handle = None
        
        if hook_handle is not None:
            hook_handle.remove()
            
        return latents_final.detach(), mask_pixel
            
    def run(self):
        timer = MyTimer()
        time_list = []
        # result_dir = self.protected_image_dir + '/' + \
        #     self.test_model_name[0] + '/' + self.target_choice
        result_dir = self.protected_image_dir + '/' + \
        self.test_model_name[0] + '/' + self.target_choice  + '/' + f"step_{self.start_step}_iter{self.prot_steps}_lr_{self.lr_emb}_adv_{self.adv_optim_weight}_wav_{self.wavelet_weight}_reg_{self.reg_weight}_lpips_{self.lpips_weight}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        target_image, _ = get_target_test_images(
            self.target_choice, self.device, self.MTCNN_cropping)
        with torch.no_grad():
            target_embeddings = self.get_FR_embeddings(target_image)

        for i, (fname, image) in enumerate(self.dataloader):
            image_name = fname[0]
            image = image.to(self.device)
            
            bb_src1 = None
            if self.MTCNN_cropping:
                path = self.source_dir + '/' + image_name + '.png'
                img = Image.open(path)
                if img.size[0] != self.image_size:
                    img = img.resize((self.image_size, self.image_size))
                bb_src1 = alignment(img)
            
            controller = AttentionControlEdit(num_steps=self.diffusion_steps,
                                              self_replace_steps=1.0)
            
            if self.is_obfuscation:
                image_hold = image.clone()
                if self.MTCNN_cropping and bb_src1 is not None:
                    h_s, h_e = round(bb_src1[1]), round(bb_src1[3])
                    w_s, w_e = round(bb_src1[0]), round(bb_src1[2])
                    if (h_e - h_s) > 0:
                        image_hold = image_hold[:, :, h_s:h_e, w_s:w_e]
                with torch.no_grad():
                    source_embeddings = self.get_FR_embeddings(image_hold)
            else:
                source_embeddings = None
            
            timer.tic()
            
            latents, mask_pixel = self.attacker(image,
                                    image_name,
                                    source_embeddings,
                                    target_embeddings,
                                    controller,
                                    None, 
                                    bb_src1)
            
            avg_time = timer.toc()
            time_list.append(avg_time)

            if latents is not None:
                self.visualize(image_name, image, latents, controller, mask_pixel=mask_pixel)
        
        print('Time: ', round(np.average(time_list), 2))
        result_fn = os.path.join(result_dir, "time.txt")
        with open(result_fn, 'a') as f:
            f.write(f"Time: {round(np.average(time_list),2)}\n")
        f.close()