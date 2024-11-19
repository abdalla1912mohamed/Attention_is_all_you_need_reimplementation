Transformer Implementation - Attention Is All You Need
This repository contains a PyTorch-based from-scratch implementation of the Transformer model, inspired by the seminal paper "Attention Is All You Need". The project demonstrates the architecture's encoder-decoder framework for tasks like sequence-to-sequence learning, machine translation, and more.

.
├── model.py        # Defines the Transformer model and its components
├── train.py        # Script to train the Transformer on a dataset
├── validation.py   # Script to validate the model and compute metrics
├── config.py       # Configuration file for hyperparameters and settings
├── dataset.py      # Dataset preparation and loading utilities
└── README.md       # Documentation (this file)

Complete Transformer Model: Implements encoder, decoder, multi-head attention, positional encoding, and feedforward layers.
Custom Training Pipeline: A modular training script for supervised learning tasks.
Validation Support: Tools to evaluate the model's performance on test datasets.
Configurable Settings: Centralized configuration via config.py for hyperparameters like learning rate, batch size, and model dimensions.
Dataset Handling: Dataset utilities for loading, tokenizing, and batching data.
Acknowledgments
This implementation is inspired by the paper "Attention Is All You Need" by Vaswani et al., and is built using PyTorch based on the youtube video : https://www.youtube.com/watch?v=ISNdQcPhsts&t=9691s
