# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.fairseq_criterion import FairseqCriterion



@register_criterion('single_shot_loss')
class SingleShotCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        # self.kl_div = torch.nn.KLDivLoss(reduction='none')
        self.mse = torch.nn.MSELoss(reduction='none')

    def label_smoothed_nll_loss(self, lprobs, target, epsilon, ignore_index=None, reduce=True):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def kldiv_loss(self, src_emb, tgt_emb, reduce=True):
        src_emb = F.log_softmax(src_emb, dim=-1).view(src_emb.size(0), -1)
        tgt_emb = tgt_emb.softmax(dim=-1).view(tgt_emb.size(0), -1)
        kldiv_loss = self.kl_div(src_emb, tgt_emb)

        if reduce:
            kldiv_loss = kldiv_loss.sum()
        return kldiv_loss


    # def mse_loss(self, src_emb, tgt_emb, reduce=True):
    #     src_emb = src_emb.view(src_emb.size(0), -1)
    #     tgt_emb = tgt_emb.view(tgt_emb.size(0), -1)
    #     mse_loss = self.mse(src_emb, tgt_emb)
    #
    #     if reduce:
    #         mse_loss = mse_loss.sum()
    #     return mse_loss


    def mse_loss(self, encoder_states, q_raws, reduce=True):
        mse_loss = 0.
        for encoder_state, q_raw in zip(encoder_states.encoder_states, q_raws):
            mse_loss += self.mse(encoder_state, q_raw).mean(dim=-1)
        if reduce:
            mse_loss = mse_loss.sum()
        return mse_loss


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, encoder_states, q_raws = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, encoder_states, q_raws, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, encoder_states, q_raws, sample, reduce=True):
        lprobs = F.log_softmax(net_output, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = self.label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        # batch_size = net_output[1].size(0)
        # kl_loss = self.kldiv_loss(net_output[1][:batch_size//2, :, :],
        #                           net_output[1][batch_size//2:, :, :])
        # loss += kl_loss
        # mse_loss = self.mse_loss(net_output[1][:batch_size // 2, :, :].contiguous(),
        #                          net_output[1][batch_size // 2:, :, :].contiguous())

        mse_loss = self.mse_loss(encoder_states, q_raws)

        loss += mse_loss
        return loss, nll_loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


