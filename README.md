# DualDiff: Dual-Variable Joint Optimization via Diffusion Model for Transferable Facial Privacy Protection




## Abstract
The widespread deployment of face recognition (FR) systems raises severe privacy concerns, as unauthorized surveillance and user trajectory tracking easily breed within social media scenarios. Relying on the powerful image generation capabilities of diffusion models, recent studies have proposed various diffusion-based adversarial face generation methods to achieve facial privacy protection. However, because diffusion models inevitably weaken adversarial perturbations during the denoising process, existing methods are constrained by the diffusion purification effect. Furthermore, the generated adversarial examples often suffer from image distortions and the destruction of background details, making it difficult to balance attack transferability and visual quality.To address these challenges, a dual-variable joint optimization (DualDiff) method is proposed. By jointly optimizing latent code and unconditional embeddings within diffusion models, DualDiff effectively expands the adversarial search space and injects adversarial perturbations, mitigating the purification effect during the diffusion process. To further ensure high visual fidelity, spatial-frequency constraints are introduced to confine perturbations to the facial region for background preservation, and to inject them into imperceptible high-frequency domains while maintaining low-frequency consistency. Extensive experiments on datasets including CelebA-HQ and LADN demonstrate that DualDiff achieves superior visual quality and significantly improves black-box transferability by 11.29% over state-of-the-art methods, while its effectiveness against commercial FR APIs (Face++ and Aliyun) is also further verified.

## Setup
- **Get code**
```shell 
git clone https://github.com/Rehmer-0131/DualDiff.git
```

- **Build environment**
```shell
cd DualDiff
# use anaconda to build environment 
conda create -n DualDiff python=3.11
conda activate DualDiff
# install packages
pip install -r requirements.txt
```

- **Download assets**
  - Download pre-trained face recognition models and datasets from [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN) and place them in the assets folder

- **The final assets folder should be like this:**
```shell
assets
  └- datasets
    └- CelebA-HQ
    └- LADN
  └- face_recognition_models
    └- facenet.pth
    └- facenet.py
    └- ...
  └- target_images
  └- test_images
```

**Run the code:**

```shell
python main.py
```

