# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq.criterions import register_criterion, label_smoothed_cross_entropy



# def predk_label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, pred_k=8):
#      flat_target = target.view(-1, 1)
#      if flat_target.dim() == lprobs.dim() - 1:
#          flat_target = flat_target.unsqueeze(-1)
#      nll_loss = -lprobs.gather(dim=-1, index=flat_target)
#
#      pred_ignore = [1 << i for i in range(int(math.log2(pred_k)))]
#      pred_ignore = torch.cumsum(torch.tensor(pred_ignore), dim=0)
#      while pred_ignore[-1] * 2 > target.shape[-1]:
#          pred_ignore = pred_ignore[:-1]
#
#      pred_k = min(pred_k, int(pred_ignore[-1]) + 1)
#
#      for k in range(1, pred_k):
#          pad_eo_target = torch.ones(target.size(0), k).long()
#
#          pad_so_stride = int(pred_ignore[0]) - 1 if k > 1 else 0
#          if pad_so_stride > 0:
#              pad_so_target = torch.ones(target.size(0), pad_so_stride).long()
#              predk_target = torch.cat([pad_so_target.to(device=target.device),
#                                      target[:, k + pad_so_stride:],
#                                      pad_eo_target.to(device=target.device)], dim=-1).view(-1, 1)
#          else:
#              predk_target = torch.cat([target[:, k:],
#                                      pad_eo_target.to(device=target.device)], dim=-1).view(-1, 1)
#
#          predk_loss = -lprobs.gather(dim=-1, index=predk_target)
#          pad_mask = predk_target.eq(ignore_index)
#          predk_loss.masked_fill_(pad_mask, 0.)
#
#          nll_loss += predk_loss  #* (1/(k + 1))
#
#          if int(pred_ignore[0]) == k:
#              pred_ignore = pred_ignore[1:]
#          if pred_ignore.numel() == 0: # In case target/pred_k is longer than the sum of the ignore steps
#              break

def predk_label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, pred_k=8):
     flat_target = target.view(-1, 1)
     if flat_target.dim() == lprobs.dim() - 1:
         flat_target = flat_target.unsqueeze(-1)
     nll_loss = -lprobs.gather(dim=-1, index=flat_target)


     pred_k = min(pred_k, int(target.shape[-1]))

     for k in range(1, pred_k):
         pad_eo_target = torch.ones(target.size(0), k).long()

         predk_target = torch.cat([target[:, k:],
                                   pad_eo_target.to(device=target.device)], dim=-1).view(-1, 1)

         predk_loss = -lprobs.gather(dim=-1, index=predk_target)
         pad_mask = predk_target.eq(ignore_index)
         predk_loss.masked_fill_(pad_mask, 0.)

         nll_loss += predk_loss * (1/(k + 1))

     smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
     nll_loss = nll_loss.squeeze(-1)
     smooth_loss = smooth_loss.squeeze(-1)
     if reduce:
         nll_loss = nll_loss.sum()
         smooth_loss = smooth_loss.sum()

     eps_i = epsilon / lprobs.size(-1)
     loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
     return loss, nll_loss


@register_criterion('predk_label_smoothed_cross_entropy')
class PredkLabelSmoothedCrossEntropyCriterion(label_smoothed_cross_entropy.LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task, sentence_avg, label_smoothing)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output) #.view(-1, 1)
        loss, nll_loss = predk_label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss


