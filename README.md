# Transformer Model for Machine Translation

This repository contains an implementation of the Transformer model for machine translation, based on the original paper "Attention is All You Need" by Vaswani et al. The model leverages self-attention mechanisms to translate text from one language to another efficiently. The code 

## Features

- **Transformer Architecture**: Incorporates the encoder-decoder structure with self-attention.
- **Configurable**: Easy customization for various source-target language pairs.
- **Greedy Decoding**: Implements a straightforward greedy decoding approach for translation.
- **PyTorch Framework**: Utilizes PyTorch for model construction, training, and inference, supporting GPU acceleration.

## Installation

Ensure Python 3.6+ is installed. Install the required libraries using pip:

```bash
pip install torch torchvision torchaudio transformers tqdm datasets
```

## Usage

### Configuration

Adjust your training and model parameters in `config.py`. This includes model architecture, training epochs, batch size, learning rate, and dataset paths.

### Training the Model

To train the Transformer model on your dataset, execute:

```bash
python train_model.py
```

The script trains the model and saves checkpoints periodically.

### Translating Text

Use the trained model to translate text from the source language to the target language. A `greedy_decode` function is provided for this purpose.

## Acknowledgements

This project was inspired by Umar Jamil's YouTube implementation [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.](https://www.youtube.com/watch?v=ISNdQcPhsts)



