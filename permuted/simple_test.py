import torch
from fairseq.data import LanguagePairDataset, TokenBlockDataset, Dictionary, ConcatDataset
from fairseq.data.concat_dataset import ConcatDataset
from unittest.mock import MagicMock, patch
from permuted_dataset import PermutedDataset


def _convert_src_tokens_to_tensor(vocab, src_tokens,
                                  append_eos):
    src_len = [len(x) for x in src_tokens]
    # If we have to append EOS, we include EOS in counting src length
    if append_eos:
        src_len = [length + 1 for length in src_len]

    x = torch.LongTensor(len(src_tokens), max(src_len)).fill_(vocab.pad())
    for i in range(len(src_tokens)):
        for j in range(len(src_tokens[i])):
            x[i][j] = vocab.index(src_tokens[i][j])
        if append_eos:
            x[i][j + 1] = vocab.eos()

    x = x.transpose(1, 0)
    return x, torch.LongTensor(src_len)


vocab = Dictionary()
vocab.add_symbol("he@@")
vocab.add_symbol("llo")
vocab.add_symbol("how")
vocab.add_symbol("are")
vocab.add_symbol("y@@")
vocab.add_symbol("ou")
vocab.add_symbol("n@@")
vocab.add_symbol("ew")
vocab.add_symbol("or@@")
vocab.add_symbol("k")

src_tokens = [
    ["he@@", "llo", "n@@", "ew", "y@@", "or@@", "k"],
    ["how", "are", "y@@", "ou"],
]
x, src_lengths = x, src_lengths = _convert_src_tokens_to_tensor(
    vocab=vocab, src_tokens=src_tokens, append_eos=False
)



d1 = vocab
d2 = vocab
token1 = x.t()
tokens_ds1 = TokenBlockDataset(
    token1,
    sizes=src_lengths,
    break_mode='complete',
    block_size=1,
    pad=0,
    eos=1,
    include_targets=False,
)
token2 = x.t()
tokens_ds2 = TokenBlockDataset(
    token2,
    sizes=src_lengths,
    break_mode='complete',
    block_size=1,
    pad=0,
    eos=1,
    include_targets=False,
)
p_tokens_ds2 = PermutedDataset(tokens_ds2, d2, seed=123)
dataset = LanguagePairDataset(
    tokens_ds1, tokens_ds1.sizes, d1,
    tokens_ds2, tokens_ds2.sizes, d2,
    shuffle=False
)
