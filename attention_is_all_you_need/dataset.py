from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Define special tokens for source and target languages
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

        # Validate special token indices
        self.vocab_size_src = tokenizer_src.get_vocab_size()
        self.vocab_size_tgt = tokenizer_tgt.get_vocab_size()
        assert self.sos_token.item() < self.vocab_size_src, "SOS token index is out of range."
        assert self.eos_token.item() < self.vocab_size_src, "EOS token index is out of range."
        assert self.pad_token.item() < self.vocab_size_src, "PAD token index is out of range."

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        # Get the source-target pair for the index
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Encode the source and target texts into token IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Debug: Print tokenized inputs
        print(f"Source tokens: {enc_input_tokens}")
        print(f"Target tokens: {dec_input_tokens}")

        # Trim the tokens if they exceed the maximum sequence length
        if len(enc_input_tokens) > self.seq_len - 2:  # Account for SOS and EOS tokens
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        if len(dec_input_tokens) > self.seq_len - 1:  # Account for SOS token
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]

        # Calculate the number of padding tokens needed for encoder and decoder
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 tokens for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 token for SOS

        # Ensure padding count is not negative
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Construct the encoder input tensor
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        # Construct the decoder input tensor (no EOS)
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Construct the label tensor (same as decoder input but with EOS token)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Debug: Check token indices against vocabulary size
        assert encoder_input.max().item() < self.vocab_size_src, f"Encoder input index out of range: {encoder_input.max().item()}"
        assert decoder_input.max().item() < self.vocab_size_tgt, f"Decoder input index out of range: {decoder_input.max().item()}"
        assert label.max().item() < self.vocab_size_tgt, f"Label index out of range: {label.max().item()}"

        # Debug: Print tensor sizes
        print(f"Encoder input size: {encoder_input.size(0)}")
        print(f"Decoder input size: {decoder_input.size(0)}")
        print(f"Label size: {label.size(0)}")

        # Ensure the tensors' sizes match the expected sequence length
        assert encoder_input.size(0) == self.seq_len, f"Encoder input size mismatch: {encoder_input.size(0)} != {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"Decoder input size mismatch: {decoder_input.size(0)} != {self.seq_len}"
        assert label.size(0) == self.seq_len, f"Label size mismatch: {label.size(0)} != {self.seq_len}"

        # Return the processed data
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    # Generate a causal mask for the decoder input to prevent attending to future tokens
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
