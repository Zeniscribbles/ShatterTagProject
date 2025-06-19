# Shatter Tag Project

Project Overview
This project is a fully reworked and generalized watermarking framework derived from StegaStamp, designed for embedding imperceptible binary fingerprints into images. Unlike traditional watermarking tools focused solely on robustness, our system is being adapted for controlled fragility — enabling it to act as a tamper seal in addition to traditional ownership verification.

We began by stripping out unnecessary diffusion model dependencies and hardcoded CIFAR-specific logic. What emerged is a clean, modular, Colab-compatible pipeline for:

Embedding binary fingerprints (watermarks) into image datasets

Decoding them reliably under benign conditions

Logging visual and scalar results across training epochs

Saving both fingerprinted images and associated bitstrings for verification
