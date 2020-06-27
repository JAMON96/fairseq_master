# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import logging
import numpy as np
import torch
from copy import deepcopy
from fairseq.data import data_utils, Dictionary, noising

from fairseq.data import BaseWrapperDataset
from fairseq.data.language_pair_dataset import LanguagePairDataset


logger = logging.getLogger(__name__)

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    if pad_idx == 'len':
        res = torch.arange(size).unsqueeze(0).repeat((len(values), 1))
    else:
        res = values[0].new(len(values), size).fill_(pad_idx)
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res



def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_idx=pad_idx):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        # if input_feeding:
        #     # we create a shifted version of targets for feeding the
        #     # previous output token(s) into the next decoder step
        #     prev_output_tokens = merge(
        #         'target',
        #         left_pad=left_pad_target,
        #         move_eos_to_beginning=True,
        #     )
        #     prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)



    permutation = merge('permutation', left_pad=False, pad_idx='len')

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': target,
            # 'src_lengths': src_lengths,
        },
        'target': permutation.index_select(0, sort_order),
    }


    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class WordShuffleWithOrder(noising.WordNoising):
    """Shuffle words by no more than k positions."""

    def __init__(self, dictionary, default_max_shuffle_distance=2**31, bpe_cont_marker="@@", bpe_end_marker=None):
        super().__init__(dictionary, bpe_cont_marker, bpe_end_marker)
        self.default_max_shuffle_distance = default_max_shuffle_distance

    def noising(self, x, lengths, max_shuffle_distance=None):
        if max_shuffle_distance is None:
            max_shuffle_distance = self.default_max_shuffle_distance

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(0,max_shuffle_distance,size=(x.size(0), x.size(1)))
        # noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = np.arange(0, x.shape[0])[:, np.newaxis] #self.get_word_idx(x)
        # word_idx = self.get_word_idx(x)
        x2 = x.clone()
        for i in range(lengths.size(0)):

            scores = word_idx[:, i] + noise[word_idx[:, i], i]

            permutation = scores.argsort()
            # shuffle words
            x2[:, i].copy_(
                x2[:, i][torch.from_numpy(permutation)]
            )
        return x2, torch.tensor(permutation)


class PredkLanguagePairDataset(LanguagePairDataset):

    def __init__(self, *kargs, **kwargs):
        self.max_shuffle_distance = kwargs.pop('max_shuffle_distance', None)
        self.augment_reverse = kwargs.pop('augment_reverse', False)
        self.relative_pos = kwargs.pop('relative_pos', False)
        self.mark_end = kwargs.pop('mark_end', False)
        super().__init__(*kargs, **kwargs)
        self.shuffler = WordShuffleWithOrder(self.tgt_dict)

    @staticmethod
    def from_pair_dataset(d, max_shuffle_distance=None, relative_pos=False, augment_reverse=False, mark_end=False):
        return PredkLanguagePairDataset(d.src, d.src_sizes, d.src_dict,
                                           tgt=d.tgt, tgt_sizes=d.tgt_sizes, tgt_dict=d.tgt_dict,
                                           left_pad_source=d.left_pad_source, left_pad_target=d.left_pad_target,
                                           max_source_positions=d.max_source_positions, max_target_positions=d.max_target_positions,
                                           shuffle=d.shuffle, input_feeding=d.input_feeding,
                                           remove_eos_from_source=d.remove_eos_from_source, append_eos_to_target=d.append_eos_to_target,
                                           align_dataset=d.align_dataset,
                                           append_bos=d.append_bos, eos=d.eos,
                                           max_shuffle_distance=max_shuffle_distance,
                                           relative_pos=relative_pos,
                                           augment_reverse=augment_reverse,
                                           mark_end=mark_end
                                           )

    def collater(self, samples):
        add_samples = []
        for s in samples:
            length = torch.tensor([s['target'].numel() - 1])
            tgt, perm = self.shuffler.noising(
                s['target'].view(-1, 1), length, max_shuffle_distance=self.max_shuffle_distance)
            s['target'] = tgt.view(-1)
            s['permutation'] = perm
            # if self.mark_end:
            #     start = length + 1
            # else:
            #     start = torch.tensor([0])
            # s['permutation'] = torch.cat(
            #     (start, perm.view(-1) + 1, length + 1)) + self.src_dict.pad() + 1
            # if self.augment_reverse and bool(torch.rand(1) > 0.5):
            #     s['target'] = torch.cat(
            #         (s['target'][:-1].flip(0), s['target'][-1:]))
            #     s['permutation'] = torch.cat(
            #         (start, perm.view(-1).flip(0) + 1, length + 1)) + self.src_dict.pad() + 1


        samples += add_samples

        x = collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

        return x
