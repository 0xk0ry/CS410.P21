# CS410 Project: Replicating "Fast is Better Than Free: Revisiting Adversarial Training"

This repository contains code and experiments for the CS410 class project, aiming to replicate the results of the paper:  
**"Fast is Better Than Free: Revisiting Adversarial Training"**  
([arXiv:2001.03994](https://arxiv.org/abs/2001.03994))

## Project Overview

The goal of this project is to reproduce the main findings of the paper, which demonstrates that fast adversarial training using FGSM (Fast Gradient Sign Method) with certain modifications can achieve robustness comparable to more expensive methods like PGD adversarial training, but with significantly reduced computational cost.

We provide scripts and configurations for running experiments on CIFAR10 and ImageNet datasets, following the methodology described in the paper.

## Repository Structure

- `fast_adversarial/`  
  Main codebase for fast adversarial training, including scripts for CIFAR10, ImageNet, and MNIST.
- `FreeAdversarialTraining/`  
  Reference implementation of Free Adversarial Training.
- `exp/`  
  Pretrained model weights and experiment outputs.
- `adversarial_experiments.ipynb`, `te.ipynb`, `test.ipynb`, `test.py`  
  Notebooks and scripts for running and analyzing experiments.

## How to Use

1. **Install Dependencies**  
   Each submodule (e.g., `fast_adversarial/CIFAR10/`, `fast_adversarial/ImageNet/`) contains its own `requirements.txt`.  
   Example for CIFAR10:
   ```bash
   cd fast_adversarial/CIFAR10
   pip install -r requirements.txt
   ```
   For mixed-precision training, install [Apex](https://github.com/NVIDIA/apex).

2. **Train a Model**  
   Example (CIFAR10, FGSM training):
   ```bash
   python train_fgsm.py --help
   ```
   For ImageNet, use the provided shell scripts in `fast_adversarial/ImageNet/` (e.g., `run_fast_2px.sh`).

3. **Evaluate a Model**  
   Use the evaluation scripts provided in each submodule to test robustness against adversarial attacks.

## References

- [Fast is Better Than Free: Revisiting Adversarial Training (arXiv:2001.03994)](https://arxiv.org/abs/2001.03994)
- [Official Fast Adversarial Training Repository](https://github.com/locuslab/fast_adversarial)
- [Free Adversarial Training Repository](https://github.com/mahyarnajibi/FreeAdversarialTraining)

## Acknowledgements

This project reuses and adapts code from the official repositories above for educational and research purposes.

---