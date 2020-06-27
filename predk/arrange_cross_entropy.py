# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn.functional as F
from fairseq.criterions import register_criterion, cross_entropy


@register_criterion('arrange_cross_entropy')
class ArrangeCrossEntropyCriterion(cross_entropy.CrossEntropyCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task, sentence_avg)

    def forward(self, model, sample, reduce=True):

        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=-100,
            reduction='none',
        )
        pad_mask = sample['net_input']['src_tokens'].view(-1).eq(1)
        loss.masked_fill_(pad_mask, 0.)
        if reduce:
            loss = loss.sum()
        return loss, loss


