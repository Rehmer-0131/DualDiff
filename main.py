# =============================================================================
# Import required libraries
# =============================================================================
import os
import random
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler


from dataset import ImageDataset

from  DualDiff_optimization import Adversarial_Opt

from testsv6 import attack_local_models


def parse_args():
    parser = argparse.ArgumentParser(description="Single-Stage DualDiff")


    parser.add_argument('--source_dir',
                        default="",
                        type=str,
                        help="source images folder path")
    parser.add_argument('--test_dir',
                        default="",
                        type=str,
                        help="test images folder path for obfuscation")
    parser.add_argument('--protected_image_dir',
                        default="results",
                        type=str)

    parser.add_argument('--comparison_null_text',
                        default=False,
                        type=bool,
                        help="If True, only compares reconstruction, skips attack")
    
    parser.add_argument('--target_choice',
                        default='2',
                        type=str,
                        help='Target identity choice')
    
    parser.add_argument("--test_model_name",
                        default=['ir152'],
                        nargs='+',
                        help="Used for naming results")
    parser.add_argument("--surrogate_model_names",
                        default=['facenet', 'irse50', 'mobile_face'],
                        nargs='+',
                        help="White-box FR models to attack")


    parser.add_argument('--MTCNN_cropping',
                        default=True,
                        type=bool)
    parser.add_argument('--is_obfuscation',
                        default=False,
                        type=bool)
    parser.add_argument('--image_size',
                        default=256, 
                        type=int)

    parser.add_argument('--diffusion_steps',
                        default=50, 
                        type=int)
    parser.add_argument('--start_step',
                        default=40, 
                        type=int,
                        help='Which inverted step to start optimization from (0-indexed)')
    
    parser.add_argument('--prot_steps',
                        default=50, 
                        type=int,
                        help='Optimization iterations for the single-stage attack')

    parser.add_argument('--lr_latent',
                        default=0.01,
                        type=float,
                        help='Learning rate for latent code optimization')
    parser.add_argument('--lr_emb',
                        default=0.01,
                        type=float,
                        help='Learning rate for unconditional embeddings optimization')


    parser.add_argument('--adv_optim_weight',
                        default=0.1, 
                        type=float,
                        help='Weight for Adversarial Loss (FR)')
    parser.add_argument('--wavelet_weight',
                        default=10.0, 
                        type=float,
                        help='wavelet')
    parser.add_argument('--semantic_weight',
                        default=0.05,
                        type=float,
                        help='Weight for Semantic Feature Divergence Loss')
    
    parser.add_argument('--lpips_weight',
                        default=1.0, 
                        type=float,
                        help='Weight for LPIPS Perceptual Loss')
    
    parser.add_argument('--reg_weight',
                        default=0.001,
                        type=float,
                        help='Weight for Embedding Regularization (preventing drift)')
    

    args = parser.parse_args()
    return args


def initialize_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    initialize_seed(seed=10)
    

    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {args.device}")
    print(f"Mode: {'Obfuscation' if args.is_obfuscation else 'Impersonation'}")
    print(f"Single-Stage Optimization: {args.prot_steps} steps")

    model_path = '/home/whu/disk1/LeiMo/stable-diffusion-2-base' 
    print(f"Loading model from: {model_path}")
    
    diff_model = StableDiffusionPipeline.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(args.device)
    
    diff_model.scheduler = DDIMScheduler.from_config(diff_model.scheduler.config)

    dataset = ImageDataset(
        args.source_dir,
        transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    )
    args.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)



    adversarial_opt = Adversarial_Opt(args, diff_model)
    adversarial_opt.run()


    print("Optimization finished. Running evaluations...")


    attack_local_models(args, protection=False)
    attack_local_models(args, protection=True)  
