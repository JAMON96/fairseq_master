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

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
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

    src_max, tgt_max = src_lengths.max(), tgt_lengths.max()
    if src_max > tgt_max:
        pad_input = torch.ones(src_lengths.size(0), src_max - tgt_max).long()
        target = torch.cat([target, pad_input], dim=1)
    elif src_max < tgt_max:
        pad_input = torch.ones(src_lengths.size(0), tgt_max - src_max).long()
        src_tokens = torch.cat([src_tokens, pad_input], dim=1)


    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            # 'src_tokens': torch.cat([src_tokens, target], dim=0),
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'target': target,
        },
        'target': target,
    }
    # if prev_output_tokens is not None:
    #     batch['net_input']['prev_output_tokens'] = prev_output_tokens
    #
    # if samples[0].get('alignment', None) is not None:
    #     bsz, tgt_sz = batch['target'].shape
    #     src_sz = batch['net_input']['src_tokens'].shape[1]
    #
    #     offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
    #     offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
    #     if left_pad_source:
    #         offsets[:, 0] += (src_sz - src_lengths)
    #     if left_pad_target:
    #         offsets[:, 1] += (tgt_sz - tgt_lengths)
    #
    #     alignments = [
    #         alignment + offset
    #         for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
    #         for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
    #         if check_alignment(alignment, src_len, tgt_len)
    #     ]
    #
    #     if len(alignments) > 0:
    #         alignments = torch.cat(alignments, dim=0)
    #         align_weights = compute_alignment_weights(alignments)
    #
    #         batch['alignments'] = alignments
    #         batch['align_weights'] = align_weights

    return batch




class SingleShotLanguagePairDataset(LanguagePairDataset):

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    @staticmethod
    def from_pair_dataset(d):
        return SingleShotLanguagePairDataset(d.src, d.src_sizes, d.src_dict,
                                             tgt=d.tgt, tgt_sizes=d.tgt_sizes, tgt_dict=d.tgt_dict,
                                             left_pad_source=d.left_pad_source, left_pad_target=d.left_pad_target,
                                             max_source_positions=d.max_source_positions,
                                             max_target_positions=d.max_target_positions,
                                             shuffle=d.shuffle, input_feeding=d.input_feeding,
                                             remove_eos_from_source=d.remove_eos_from_source,
                                             append_eos_to_target=d.append_eos_to_target,
                                             align_dataset=d.align_dataset,
                                             append_bos=d.append_bos, eos=d.eos,
                                             )

    def collater(self, samples):

        x = collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

        return x
