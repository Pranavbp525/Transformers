import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Greedily decodes a sequence of tokens from the model.

    Parameters:
    - model (nn.Module): The transformer model for sequence-to-sequence learning.
    - source (Tensor): The input tensor containing source sequence tokens.
    - source_mask (Tensor): The mask tensor for the source input.
    - tokenizer_src (Tokenizer): The tokenizer for the source language.
    - tokenizer_tgt (Tokenizer): The tokenizer for the target language.
    - max_len (int): The maximum length of the output sequence.
    - device (torch.device): The device on which to perform the computation.

    Returns:
    - Tensor: The decoded sequence of tokens.
    """
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_inp = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_inp.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_inp.size(1)).type_as(source_mask).to(device)
        out = model.decode(decoder_inp, encoder_output, source_mask, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=-1)
        decoder_inp = torch.cat([decoder_inp, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                                dim=1)

        if next_word == eos_idx:
            break

    return decoder_inp.squeeze(0)


def get_all_sentences(ds, lang):
    """
    Generator that yields all sentences from a dataset for a given language.

    Parameters:
    - ds (Dataset): The dataset from which to extract sentences.
    - lang (str): The language code of the sentences to yield.

    Yields:
    - str: A sentence in the specified language.
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves or trains a tokenizer based on the given dataset and language.

    Parameters:
    - config (dict): Configuration dictionary containing tokenizer file paths.
    - ds (Dataset): The dataset to use for training the tokenizer.
    - lang (str): The language for which to build or get the tokenizer.

    Returns:
    - Tokenizer: The tokenizer for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Prepares the training and validation datasets.

    Parameters:
    - config (dict): Configuration dictionary specifying dataset parameters.

    Returns:
    - DataLoader: The DataLoader for the training dataset.
    - DataLoader: The DataLoader for the validation dataset.
    - Tokenizer: The tokenizer for the source language.
    - Tokenizer: The tokenizer for the target language.
    """
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def train_model(config):
    """
    Trains the model based on the given configuration.

    Parameters:
    - config (dict): Configuration dictionary containing training parameters.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'],
                        config['seq_len'], config['num_layers'], config['d_model']).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch: {epoch:02d}')

        for batch in batch_iterator:
            encoder_inp = batch['encoder_input'].to(device)
            decoder_inp = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            enocder_output = model.encode(encoder_inp, encoder_mask)
            decoder_output = model.decode(decoder_inp, enocder_output, encoder_mask, decoder_mask)
            output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state,
                   writer, num_examples=2):
    """
    Runs the validation step, printing out examples of translations.

    Parameters:
    - model (nn.Module): The transformer model.
    - validation_ds (DataLoader): DataLoader for the validation dataset.
    - tokenizer_src (Tokenizer): The tokenizer for the source language.
    - tokenizer_tgt (Tokenizer): The tokenizer for the target language.
    - max_len (int): Maximum length of the decoded sequences.
    - device (torch.device): The device to run the validation on.
    - print_msg (function): Function to print messages.
    - global_state (int): Global step count for logging.
    - writer (SummaryWriter): TensorBoard writer for logging.
    - num_examples (int): Number of examples to print during validation.
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_inp = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_inp.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_inp, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            predicted_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(predicted_text)

            print_msg('-' * 80)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{predicted_text}")

            if count == num_examples:
                print_msg('-' * 80)
                break


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
