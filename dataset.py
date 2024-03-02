import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def causal_mask(size):
    """
    Creates a square causal mask to mask out future tokens in a sequence. This is used in the attention mechanism
    to prevent the model from looking ahead into future tokens.

    Parameters:
    - size (int): The size of the square matrix, typically the sequence length.

    Returns:
    - torch.Tensor: A boolean mask where True values allow attention and False values prevent it, ensuring that
      a token cannot attend to tokens in the future.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    """
    A PyTorch Dataset for loading bilingual sentence pairs.

    Parameters:
    - ds (Dataset): A dataset containing bilingual sentence pairs.
    - tokenizer_src (Tokenizer): Tokenizer for the source language.
    - tokenizer_tgt (Tokenizer): Tokenizer for the target language.
    - src_lang (str): Source language code.
    - tgt_lang (str): Target language code.
    - seq_len (int): Maximum sequence length. Sentences longer than this length will cause an error.


    """
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_inp_tokens = self.tokenizer_src.encode(src_text).ids
        dec_inp_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_pad_tokens = self.seq_len - len(enc_inp_tokens) - 2
        dec_num_pad_tokens = self.seq_len - len(dec_inp_tokens) - 1

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_inp_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_inp_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_inp_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
