from pathlib import Path


def get_config():
    """
    Generates a configuration dictionary with training parameters and paths for model training.

    Returns:
    - dict: A dictionary containing training parameters and paths. Keys include "batch_size", "num_epochs",
      "lr" (learning rate), "seq_len" (sequence length), "d_model" (dimension of the model),
      "num_layers" (number of encoder and decoder layers), "lang_src" (source language code),
      "lang_tgt" (target language code), "model_folder" (directory to save model weights),
      "model_basename" (base name for saved model files), "preload" (path to preloaded model weights, if any),
      "tokenizer_file" (template path for tokenizer files), and "experiment_name" (name for the experiment, used
      for logging and saving models).
    """
    return {
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        'num_layers': 6,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/t_model"
    }


def get_weights_file_path(config, epoch):
    """
    Constructs the file path for saving or loading a model checkpoint for a given epoch.

    Parameters:
    - config (dict): Configuration dictionary containing "model_folder" and "model_basename".
    - epoch (int or str): The epoch number or identifier for the model checkpoint.

    Returns:
    - str: The full path to the model checkpoint file.
    """
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
