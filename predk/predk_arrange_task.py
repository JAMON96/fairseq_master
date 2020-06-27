import logging
import numpy as np
import torch
from torch import Tensor
import types
from typing import Optional, Dict
import math
from fairseq import search, checkpoint_utils
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel, search
from fairseq.data import data_utils, FairseqDataset, iterators
from fairseq.tasks import register_task
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.tasks.translation import TranslationTask
logger = logging.getLogger(__name__)



class PredkArrangeSequenceGenerator(SequenceGenerator):

    def __init__(self, models, tgt_dict, pred_k, **kwargs):
        super().__init__(models, tgt_dict, **kwargs)
        self.model = PredkArrangeEnsembleModel(models)
        self.pred_k = pred_k
        self.reorder_model, _ = checkpoint_utils.load_model_ensemble(
            ["/home/bweinstein/algo_git/Algorithms/fairseq_master/results/checkpoints/predk_reorder_16/checkpoint_best.pt"])
        self.reorder_model = self.reorder_model[0].cuda()

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)


    @torch.no_grad()
    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        input_size = src_tokens.size()
        bsz, src_len = input_size[0], input_size[1]
        beam_size = self.beam_size

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never choosed for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        # list of completed sentences
        finalized = [[] for i in range(bsz)]

        num_pred_tokens = 0
        pred_steps = [1 << i for i in range(int(math.log2(self.pred_k)))][1:]
        while sum(pred_steps) < max_len + 1:
            left = min(max_len + 1 - sum(pred_steps), pred_steps[-1])
            pred_steps.append(left)

        while num_pred_tokens < max_len + 1:

            lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, :num_pred_tokens + 1], encoder_outs, self.temperature
            )


            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            curr_step = pred_steps.pop(0)
            pred_scores, pred_indices = lprobs.topk(curr_step, -1)

            tokens[:, num_pred_tokens + 1: num_pred_tokens + curr_step + 1] = pred_indices
            scores[:, num_pred_tokens: num_pred_tokens + curr_step] = pred_scores

            num_pred_tokens += curr_step

        tokens = torch.cat([tokens[:, 1:], tokens[:, 0].unsqueeze(-1)], dim=-1)
        # pred_out = self.reorder_model.forward(tokens)
        #
        # ts, ti = pred_out.topk(1, -1)
        # ts = ts.squeeze(-1)
        # ti = ti.squeeze(-1)

        for i, sent in enumerate(tokens):
            # _, sorted_indices = torch.sort(ti[i])
            # sorted_sent = sent[sorted_indices]

            eos_pos = (sent == self.eos).nonzero()[0]
            sent = sent[: eos_pos + 1]

            hyp = {
                'tokens': sent,
                'score': scores[i].mean(),
                'attention': None,
                'alignment': None,
                'positional_scores': scores[i],
            }
            finalized[i].append(hyp)


        return finalized




@register_task('predk_arrange_translation')
class PredkArrangeTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        super(PredkArrangeTranslationTask, PredkArrangeTranslationTask).add_args(parser)
        parser.add_argument('--max-shuffle-distance', default=16, type=int,
                            help='max distance between shuffled words')
        parser.add_argument('--augment-reverse', action='store_true', default=False,
                            help='augment with reverese permutation')
        parser.add_argument('--no-relative-pos', action='store_true', default=False,
                            help='relative prediction position')
        parser.add_argument('--mark-end', action='store_true', default=False,
                            help='mark end position')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.max_shuffle_distance = args.max_shuffle_distance
        self.augment_reverse = args.augment_reverse
        self.mark_end = args.mark_end
        self.relative_pos = not args.no_relative_pos


    def build_generator(self, models, args):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = PredkArrangeSequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            pred_k=getattr(args, "predk", 8),
            beam_size=getattr(args, "beam", 1),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 30),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
        )


class PredkArrangeEnsembleModel(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)
        self.incremental_states = None
        self.has_incremental = False