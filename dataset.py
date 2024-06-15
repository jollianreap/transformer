from typing import Any

import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt,
                 src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        # Convert tokens into array of ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # Define number of padding tokens to fill sentences
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 because there're start of sentence and end of sentence tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        # Padding tokens should never be less than zero
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.eos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        # label is what we except as output from the encoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            # (1, 1, seq_len) DEBUG THIS LINE
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # we "turn off" all pad tokens in self-attention mechanism
            # (1, seq_len) & (1, seq_Len, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # here we only want words which are before it and not PAD to participate in self-attention
            'label': label, # (seq_len),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def casual_mask(size):
    """
    That is the method which gets matrix like this:
        [1, 2, 3, 4, 5, 6]
        [1, 2, 3, 4, 5, 6]
        [1, 2, 3, 4, 5, 6]
        [1, 2, 3, 4, 5, 6]
        [1, 2, 3, 4, 5, 6]
        [1, 2, 3, 4, 5, 6]

    And masks all values above diagonal in that way:
        [1, 0, 0, 0, 0, 0]
        [1, 2, 0, 0, 0, 0]
        [1, 2, 3, 0, 0, 0]
        [1, 2, 3, 4, 0, 0]
        [1, 2, 3, 4, 5, 0]
        [1, 2, 3, 4, 5, 6]

    :return: True, False states for each element in matrix
    """
    # torch.triu returns all values above the diagonal
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
