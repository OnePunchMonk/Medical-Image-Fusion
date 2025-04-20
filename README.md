# Multi-Modal CT–MRI Image Fusion

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Tech Stack](#tech-stack)
- [Future Enhancements](#future-enhancements)

## Introduction
This repository implements a deep learning pipeline for fusing computed tomography (CT) and magnetic resonance imaging (MRI) scans into a single, information-rich image. By combining anatomical detail from CT with soft-tissue contrast from MRI, the fused output supports enhanced visualization for clinical diagnosis and research applications.

## Problem Statement
Medical image fusion integrates complementary modalities—CT provides high spatial resolution, while MRI reveals soft-tissue structures—into a unified representation. This multimodal integration aims to reduce information loss and artifacts common in traditional fusion techniques by learning joint feature representations directly from the data.

## Dataset
We use the **Harvard Whole Brain Atlas**, which offers co-registered CT, MRI, and SPECT scans across normal and pathological cases. Paired CT–MRI slices are preprocessed (normalized and resized) and split into training, validation, and test sets to ensure robust evaluation.

## Model Architecture
1. **Dual-Stream Encoders**: Separate convolutional encoders extract hierarchical features from CT and MRI inputs using Conv–BatchNorm–ReLU blocks and max-pooling.
2. **Fusion Module**: Deep features from both encoders are concatenated and refined with 3×3 convolutions to promote cross-modal complementarity.
3. **Kolmogorov–Arnold Module (KAM)**: Inspired by the Kolmogorov–Arnold representation theorem, this module approximates multivariate transforms by summing univariate functions via multiple 1×1 “ψ” projections to scalars, followed by “φ” expansions back to the feature dimension, with ReLU activations.
4. **Decoder**: A transposed-convolutional decoder upsamples the fused features through ConvBlock layers and outputs a single-channel fused image via a 1×1 convolution.

## Evaluation Metrics
- **Structural Similarity Index (SSIM)**: Measures perceptual similarity (luminance, contrast, structure) between fused and reference images, ranging from 0 to 1 (higher is better).
- **Peak Signal-to-Noise Ratio (PSNR)**: Quantifies reconstruction fidelity in decibels; higher values indicate lower mean-squared error.

## Tech Stack

| Component               | Technology                       |
|-------------------------|----------------------------------|
| Programming Language    | Python 3.x                       |
| Deep Learning Framework | PyTorch 2.x                      |
| Image Processing        | OpenCV, PIL                      |
| Visualization           | Matplotlib                       |

## Future Enhancements
- **Image Denoising**: Integrate denoising modules or adversarial losses to suppress noise, especially in low-dose CT scans.
- **Additional Modalities**: Extend to PET–MRI and tri-modal (CT, MRI, SPECT) fusion for richer diagnostic insights.
- **Model Compression**: Apply pruning or quantization to enable real-time inference on edge devices.

