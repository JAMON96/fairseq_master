# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from copy import deepcopy
from fairseq.data import data_utils, Dictionary, noising

from fairseq.data import BaseWrapperDataset
from fairseq.data.language_pair_dataset import LanguagePairDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    """ Almost identical to language_pair_dataset/collate with addition of permutation"""
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning(
                "alignment size mismatch found, skipping alignment!")
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
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True)
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
        tgt_lengths = torch.LongTensor(
            [s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('permutation', None) is not None:
        permutation = merge('permutation', left_pad=False)
        permutation = permutation.index_select(0, sort_order)
        batch['net_input']['pred_positions'] = permutation[:, 1:]
        batch['net_input']['prev_positions'] = permutation[:, :-1]

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order),
                                       dtype=torch.long) * tgt_sz)
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
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0), x.size(1)),
        )
        # noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        # idx_list = list()
        # div = x.shape[0] // max_shuffle_distance
        # left = x.shape[0] % max_shuffle_distance
        # for i in range(0, div):
        #     idx_list.append(np.ones(max_shuffle_distance) * (i * max_shuffle_distance + 1))
        # idx_list.append(np.ones(left) * (div * max_shuffle_distance + 1))
        # word_idx = np.concatenate(idx_list)[:, np.newaxis]
        word_idx = np.arange(0, x.shape[0])[:, np.newaxis] #self.get_word_idx(x)
        x2 = x.clone()
        for i in range(lengths.size(0)):
            length_no_eos = lengths[i]
            if x[lengths[i] - 1, i] == self.dictionary.eos():
                length_no_eos = lengths[i] - 1
            # generate a random permutation
            scores = word_idx[:length_no_eos, i] + \
                noise[word_idx[:length_no_eos, i], i]
            # scores = word_idx[:length_no_eos, i] + \
            #          noise[:length_no_eos, i]
            # ensure no reordering inside a word
            # scores += 1e-6 * np.arange(length_no_eos)
            permutation = scores.argsort()
            # shuffle words
            x2[:length_no_eos, i].copy_(
                x2[:length_no_eos, i][torch.from_numpy(permutation)]
            )
        return x2, torch.tensor(permutation)


class PermutedLanguagePairDataset(LanguagePairDataset):

    def __init__(self, *kargs, **kwargs):
        self.max_shuffle_distance = kwargs.pop('max_shuffle_distance', None)
        self.augment_reverse = kwargs.pop('augment_reverse', False)
        self.relative_pos = kwargs.pop('relative_pos', False)
        self.mark_end = kwargs.pop('mark_end', False)
        super().__init__(*kargs, **kwargs)
        self.shuffler = WordShuffleWithOrder(self.tgt_dict)

    @staticmethod
    def from_pair_dataset(d, max_shuffle_distance=None, relative_pos=False, augment_reverse=False, mark_end=False):
        return PermutedLanguagePairDataset(d.src, d.src_sizes, d.src_dict,
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
            if self.mark_end:
                start = length + 1
            else:
                start = torch.tensor([0])
            s['permutation'] = torch.cat(
                (start, perm.view(-1) + 1, length + 1)) + self.src_dict.pad() + 1
            if self.augment_reverse and bool(torch.rand(1) > 0.5):
                s['target'] = torch.cat(
                    (s['target'][:-1].flip(0), s['target'][-1:]))
                s['permutation'] = torch.cat(
                    (start, perm.view(-1).flip(0) + 1, length + 1)) + self.src_dict.pad() + 1

            # if self.augment_reverse:
            #     rs = deepcopy(s)
            #     rs['target'] = torch.cat(
            #         (rs['target'][:-1].flip(0), rs['target'][-1:]))
            #     rs['permutation'] = torch.cat(
            #         (start, length + 1, perm.view(-1).flip(0) + 1)) + self.src_dict.pad() + 1
            #     add_samples.append(rs)

        samples += add_samples

        x = collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

        try:
            if self.relative_pos:
                x['net_input']['pred_positions'] = \
                    x['net_input']['pred_positions'] - \
                    x['net_input']['prev_positions']
        except:
            pass
        return x
