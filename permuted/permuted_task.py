import logging
import math
import torch
from torch import Tensor

import types
from typing import Optional, Dict, List
from fairseq import utils


from fairseq.data import data_utils, FairseqDataset, iterators
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel, search

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.models.fairseq_encoder import EncoderOut

from .permuted_dataset import PermutedLanguagePairDataset
logger = logging.getLogger(__name__)




class PermutedSequenceGenerator(SequenceGenerator):
    def __init__(self, models, tgt_dict, pred_k, **kwargs):
        super().__init__(models, tgt_dict, **kwargs)
        self.pos_k = pred_k


    def replicate_encoder_out_pos_k(self, encoder_out, T):

        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.repeat(1, T, 1)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.repeat(T, 1)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.repeat(T, 1, 1)
        )
        src_tokens = encoder_out.src_tokens
        src_lengths = encoder_out.src_lengths
        encoder_states = encoder_out.encoder_states

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )


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
        self.model = PermutedEnsembleModel(models)
        return self._generate(self.model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
            self,
            model,
            sample,
            prefix_tokens=None,
            bos_token=None,
            **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None
        output_pos = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(-1)
        output_pos_buf = output_pos.clone()
        output_pos[:, 0] = 2  # First position is 2

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        prev_positions = tokens.narrow(1, 0, 1).contiguous()
        T_pred = 1  # min(max_len, self.pos_k)
        pred_full = torch.arange(3, max_len + 4)
        pred_positions = pred_full[:T_pred] \
            .contiguous().view(-1, 1).repeat(1, bsz).view(-1, 1).to(device=prev_positions.device)
        # pred_positions[0, : curr_pos_k].view(-1, 1).repeat(1, bsz).view(-1,
        #                                                                 1).contiguous()


        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def mask_words_start(last_tokens, lprobs, T):
            # start_ii = torch.tensor(np.array([
            #     self.tgt_dict[t].endswith("@@")
            #     for t in last_tokens
            # ]))
            #
            # if start_ii.sum() > 0:
            #     lprobs[start_ii, 0, :] = lprobs[start_ii, 0, :].masked_fill_(self.words_start.view(1, -1), -math.inf)

            # penalize word repeat
            lprobs[:, 0].scatter_(-1, last_tokens.unsqueeze(-1),
                                  last_tokens.unsqueeze(-1).clone().float().fill_(-math.inf))

            for ts in range(T - 1):
                _, top_ind = lprobs[:, ts, :].topk(1, dim=-1)

                if top_ind.numel() > 0:
                    #     start_ii = torch.tensor(np.array([
                    #         self.tgt_dict[t].endswith("@@")
                    #         for t in top_ind.squeeze()
                    #     ]))
                    #
                    #     if start_ii.sum() > 0:
                    #         lprobs[start_ii, ts + 1, :] = lprobs[start_ii, ts + 1, :].masked_fill_(self.words_start.view(1, -1),
                    #                                                                      -math.inf)
                    # penalize word repeat
                    lprobs[:, ts + 1].scatter_(-1, top_ind, top_ind.clone().float().fill_(-math.inf))
                    lprobs[:, ts + 2].scatter_(-1, top_ind, top_ind.clone().float().fill_(-math.inf))

            return lprobs

        def finalize_hypos_for_pos_k(step, sorted_tokens, sorted_scores, eos_scores, bbsz_idx):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """

            # clone relevant token and attention tensors
            # tokens_clone = sorted_tokens
            # assert not tokens_clone.eq(self.eos).any()
            # tokens_clone[:, step] = self.eos
            # attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # compute scores per token position
            pos_scores = sorted_scores.index_select(0, bbsz_idx)[:, :step + 1]
            # pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            # cum_unfin = []
            # prev = 0
            # for f in finished:
            #     if f:
            #         prev += 1
            #     else:
            #         cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                # unfin_idx = idx // beam_size
                #
                # sent = unfin_idx + cum_unfin[unfin_idx]
                #
                # sents_seen.add((sent, unfin_idx))
                #
                # if self.match_source_len and step > src_lengths[unfin_idx]:
                #     score = -math.inf

                def get_hypo():
                    # if attn_clone is not None:
                    #     # remove padding tokens from attn scores
                    #     hypo_attn = attn_clone[i]
                    # else:
                    #     hypo_attn = None

                    return {
                        'tokens': sorted_tokens[i],
                        'score': score,
                        # 'attention': hypo_attn,  # src_len x tgt_len
                        'attention': None,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                finalized[idx].append(get_hypo())

            # newly_finished = []
            # for sent, unfin_idx in sents_seen:
            #     # check termination conditions for this sentence
            #     if not finished[sent] and is_finished(sent, step, unfin_idx):
            #         finished[sent] = True
            #         newly_finished.append(unfin_idx)
            # return newly_finished

        reorder_state = None
        batch_idxs = None
        # max_len = output_pos[:, 0].max() - 3
        for step in range(max_len // self.pos_k + 2):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # if reorder_state is not None:
            #     if batch_idxs is not None:
            #         # update beam indices to take into account removed sentences
            #         corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
            #         reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
            #     model.reorder_incremental_state(reorder_state)
            #     encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

            # curr_pred_positions = pred_positions[0, : curr_pos_k].view(-1, 1).repeat(1, bsz).view(-1,
            #                                                        1).contiguous()

            T_tokens = prev_positions.shape[-1]
            curr_pos_k = 1 if step == 0 else min(self.pos_k, max_len - T_tokens + 1)
            if curr_pos_k <= 0 or pred_positions.numel() < 1:
                break
            perm_encoder_outs = [self.replicate_encoder_out_pos_k(encoder_outs[0], curr_pos_k)]

            # if step > 0:
            #     pred_positions[:, 0] = prev_positions.repeat(curr_pos_k, 1)[:, 0]

            lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :T_tokens].repeat(curr_pos_k, 1),
                prev_positions.repeat(curr_pos_k, 1),
                pred_positions - prev_positions.repeat(curr_pos_k, 1),
                self.pos_k,
                perm_encoder_outs,
                temperature=self.temperature,
            )

            lprobs = lprobs.view(-1, bsz, self.vocab_size).transpose(0, 1)
            avg_attn_scores = avg_attn_scores[:, -1, :].view(-1, bsz, avg_attn_scores.size(-1)).transpose(0, 1)

            lprobs[lprobs != lprobs] = -math.inf

            lprobs[:, :, self.pad] = -math.inf  # never select pad
            lprobs[:, :, self.unk] -= self.unk_penalty  # apply unk penalty
            # lprobs[:, :, 5] -= self.unk_penalty  # period penalty

            if step > 0:
                try:
                    # _, last_pos = prev_positions.topk(1, -1)
                    # last_token = tokens.gather(1, last_pos)
                    lprobs = mask_words_start(tokens[:, T_tokens - 1], lprobs, curr_pos_k)
                except:
                    pass

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :, self.eos] = -math.inf
                lprobs[:, :, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            # if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
            #     prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
            #     prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
            #     prefix_mask = prefix_toks.ne(self.pad)
            #     lprobs[prefix_mask] = -math.inf
            #     lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
            #         -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
            #     )
            #     # if prefix includes eos, then we should make sure tokens and
            #     # scores are the same across all beams
            #     eos_mask = prefix_toks.eq(self.eos)
            #     if eos_mask.any():
            #         # validate that the first beam matches the prefix
            #         first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
            #         eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            #         target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            #         assert (first_beam == target_prefix).all()
            #
            #         def replicate_first_beam(tensor, mask):
            #             tensor = tensor.view(-1, beam_size, tensor.size(-1))
            #             tensor[mask] = tensor[mask][:, :1, :]
            #             return tensor.view(-1, tensor.size(-1))
            #
            #         # copy tokens, scores and lprobs from the first beam to all beams
            #         tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
            #         scores = replicate_first_beam(scores, eos_mask_batch_dim)
            #         lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            # elif step < self.min_len:
            #     # minimum length constraint (does not apply if using prefix_tokens)
            #     lprobs[:, :, self.eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                cpu_tokens = tokens.cpu()
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = cpu_tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        if ngram[-1] != self.pad:
                            gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, avg_attn_scores.size(-1), max_len + 2)
                    attn_buf = attn.clone()
                # if self.pos_k > 1:
                attn[:, :, :avg_attn_scores.size(1)].copy_(
                    avg_attn_scores.transpose(1, 2))  # TODO: Handle fising the scores after top pos_k
                # else:
                #     attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(cpu_tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    banned_tokens_per_sample = gen_ngrams[bbsz_idx].get(ngram_index, [])
                    banned_tokens_per_sample = [(bbsz_idx, t) for t in banned_tokens_per_sample]
                    return banned_tokens_per_sample

                banned_tokens = []
                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    for bbsz_idx in range(bsz * beam_size):
                        banned_tokens.extend(calculate_banned_tokens(bbsz_idx))

                if banned_tokens:
                    banned_tokens = torch.LongTensor(banned_tokens)
                    lprobs.index_put_(tuple(banned_tokens.t()), lprobs.new_tensor([-math.inf] * len(banned_tokens)))

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                beam_size,
                scores.view(bsz, beam_size, -1)[:, :, :step * self.pos_k],
            )

            # if cand_indices.size(1) != curr_pos_k:
            # cand_scores, idxs = cand_scores.topk(curr_pos_k, dim=1)
            # cand_indices = cand_indices.gather(1, idxs).view(bsz, curr_pos_k, -1)

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams[:, 0, :].add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            # eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            # eos_mask[:, :beam_size][blacklist] = 0
            eos_mask = torch.zeros_like(blacklist).type(torch.bool)
            #
            # # only consider eos when it's among the top beam_size indices
            # torch.masked_select(
            #     cand_bbsz_idx[:, :beam_size],
            #     mask=eos_mask[:, :beam_size],
            #     out=eos_bbsz_idx,
            # )
            #
            # finalized_sents = set()
            # if eos_bbsz_idx.numel() > 0:
            #     torch.masked_select(
            #         cand_scores[:, :beam_size],
            #         mask=eos_mask[:, :beam_size],
            #         out=eos_scores,
            #     )
            #     finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
            #     num_remaining_sent -= len(finalized_sents)
            #
            # assert num_remaining_sent >= 0
            # if num_remaining_sent == 0:
            #     break
            # assert step < max_len
            #
            # if len(finalized_sents) > 0:
            #     new_bsz = bsz - len(finalized_sents)
            #
            #     # construct batch_idxs which holds indices of batches to keep for the next pass
            #     batch_mask = cand_indices.new_ones(bsz)
            #     batch_mask[cand_indices.new(finalized_sents)] = 0
            #     batch_idxs = batch_mask.nonzero().squeeze(-1)
            #
            #     eos_mask = eos_mask[batch_idxs]
            #     cand_beams = cand_beams[batch_idxs]
            #     bbsz_offsets.resize_(new_bsz, 1)
            #     cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            #     cand_scores = cand_scores[batch_idxs]
            #     cand_indices = cand_indices[batch_idxs]
            #     if prefix_tokens is not None:
            #         prefix_tokens = prefix_tokens[batch_idxs]
            #     src_lengths = src_lengths[batch_idxs]
            #     blacklist = blacklist[batch_idxs]
            #
            #     scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            #     scores_buf.resize_as_(scores)
            #     tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            #     tokens_buf.resize_as_(tokens)
            #     if attn is not None:
            #         attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
            #         attn_buf.resize_as_(attn)
            #     bsz = new_bsz
            # else:
            batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )

            if step == 0:
                active_scores = torch.gather(
                    cand_scores, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=scores[:, 0].view(bsz, -1, beam_size),
                )
            else:
                active_scores = torch.gather(
                    cand_scores, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=scores[:, T_tokens - 1:T_tokens + self.pos_k - 1].view(bsz, -1, beam_size),
                )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            # active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :T_tokens], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :T_tokens],
            )

            if step == 0:
                torch.gather(
                    cand_indices, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=tokens_buf[:, 1].view(bsz, -1, beam_size),
                )
            else:
                torch.gather(
                    cand_indices, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=tokens_buf.view(bsz, -1, beam_size)[:, T_tokens:T_tokens + curr_pos_k, :],
                )

            # copy output permuted positions
            torch.index_select(
                output_pos[:, :T_tokens], dim=0, index=active_bbsz_idx,
                out=output_pos_buf[:, :T_tokens],
            )

            # if cand_indices.size(1) != curr_pos_k:
            # cand_pos = torch.gather(
            #     idxs, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
            # )
            #
            # cand_pos = pred_positions[:, -1].view(-1, bsz, 1).transpose(0, 1).gather(1, cand_pos)
            # else:
            cand_pos = pred_positions[:, -1].view(-1, bsz, 1).transpose(0, 1)

            if step == 0:
                torch.gather(
                    cand_pos, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=output_pos_buf[:, 1].view(bsz, -1, beam_size),
                )
            else:
                torch.gather(
                    cand_pos, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=output_pos_buf.view(bsz, -1, beam_size)[:, T_tokens:T_tokens + curr_pos_k, :],
                )

            # copy scores
            if step > 0:
                torch.index_select(
                    scores[:, :T_tokens], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :T_tokens],
                )
                torch.gather(
                    cand_scores, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=scores_buf.view(bsz, -1, beam_size)[:, T_tokens - 1:T_tokens + curr_pos_k - 1],
                )
            else:
                torch.gather(
                    cand_scores, dim=-1, index=active_hypos.unsqueeze(-1).repeat(1, curr_pos_k, 1),
                    out=scores_buf[:, 0].view(bsz, -1, beam_size),
                )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step * self.pos_k + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step * self.pos_k + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            output_pos, output_pos_buf = output_pos_buf, output_pos
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

            prev_positions = output_pos[:, :2] if step == 0 else output_pos[:, :T_tokens + curr_pos_k]
            pred_list = list()
            for b in range(bsz):
                pred_samp = sorted(set(pred_full.tolist()) - set(output_pos[b].tolist()) - {-1})
                pred_samp = torch.tensor(list(pred_samp)).to(dtype=torch.long, device=pred_positions.device)
                # perm_size = min(4, pred_samp.size(0), output_pos[b, 0] - (prev_positions.size(-1) + 2))
                # if perm_size > 1:
                #     pred_samp[:perm_size] = pred_samp[torch.randperm(perm_size)]
                pred_list.append(pred_samp.unsqueeze(0))
                # pred_list.append(pred_samp[torch.randperm(pred_samp.size()[0])].unsqueeze(0))
            pred_positions = torch.cat(pred_list)
            # T = min(self.pos_k, pred_positions.shape[1])
            T = min(self.pos_k, max_len - prev_positions.shape[1] + 1)
            # pred_positions = (prev_positions - 1)[:, -T:]
            pred_positions = pred_positions[:, :T].transpose(0, 1).contiguous().view(-1, 1)
            prev_buf = prev_positions[:, 1:].repeat(T, 1)
            pred_positions = torch.cat([prev_buf, pred_positions], dim=1)

        output_pos.masked_fill_(output_pos < 3, 999999)
        _, sorted_indices = torch.sort(output_pos - 2)
        sorted_tokens = torch.cat([tokens[b][sidx].unsqueeze(0) for b, sidx in enumerate(sorted_indices)])
        sorted_scores = torch.cat([scores[b][sidx].unsqueeze(0) for b, sidx in enumerate(sorted_indices[:, 1:] - 1)])

        for pos in range(max_len + 1):
            eos_mask = sorted_tokens[:, pos + 1].eq(self.eos)
            torch.masked_select(
                torch.arange(0, bsz).cuda(),
                mask=eos_mask,
                out=eos_bbsz_idx,
            )
            finished_sent = list()
            if eos_bbsz_idx.numel() > 0:
                for idx, sent in enumerate(finalized):
                    if sent:
                        eos_mask[idx] = 0
                        finished_sent.append(idx)
                        eos_bbsz_idx = eos_bbsz_idx[eos_bbsz_idx != idx]
                if len(finished_sent) == bsz:
                    break
                if torch.sum(eos_mask) == 0:
                    continue
                finalize_hypos_for_pos_k(pos, sorted_tokens[eos_bbsz_idx, :pos + 2],
                                         sorted_scores, sorted_scores[eos_bbsz_idx, pos], eos_bbsz_idx)

        finished_sent = list()
        for idx, sent in enumerate(finalized):
            if sent:
                finished_sent.append(idx)
        if len(finished_sent) < bsz:
            not_finished_idx = torch.tensor(list(set(range(bsz)) - set(finished_sent)), device=tokens.device)
            sorted_tokens[not_finished_idx, -1] = self.eos
            finalize_hypos_for_pos_k(max_len + 1, sorted_tokens[not_finished_idx, 1:],
                                     sorted_scores, sorted_scores[not_finished_idx, -1], not_finished_idx)

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        return finalized




    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        Returns number of sentences being finalized.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            unfin_idx = idx // beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))
            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether we've finished generation for a given sentence, by
        comparing the worst score among finalized hypotheses to the best
        possible score among unfinalized hypotheses.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(
        self,
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf, dtype=torch.float)
        return lprobs




class PermutedEnsembleModel(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)
        self.incremental_states = None

    @torch.no_grad()
    def forward_decoder(self, tokens, prev_positions, pred_positions, pos_k, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                prev_positions,
                pred_positions,
                pos_k,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                prev_positions,
                pred_positions,
                pos_k,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn


    def _decode_one(
            self, tokens, prev_positions,
            pred_positions,
            pos_k, model, encoder_out,
            incremental_states, log_probs,
            temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                prev_positions=prev_positions,
                pred_positions=pred_positions,
                pos_k=pos_k,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(
                tokens,
                prev_positions=prev_positions,
                pred_positions=pred_positions,
                encoder_out=encoder_out,
            ))
        # t_out = decoder_out[0].shape[1] if pos_k > 1 else 1
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1] if len(decoder_out) > 1 else None
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1:, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1:, :]
        return probs, attn





@register_task('permuted_translation')
class PermutedTranslationTask(TranslationTask):
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
        super(PermutedTranslationTask, PermutedTranslationTask).add_args(parser)
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

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch=1, combine=False, **kwargs)
        # if split == 'train':
        self.datasets[split] = PermutedLanguagePairDataset.from_pair_dataset(
            self.datasets[split],
            max_shuffle_distance=self.max_shuffle_distance,
            augment_reverse=self.augment_reverse,
            relative_pos=self.relative_pos,
            mark_end=self.mark_end)



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
            search_strategy = PermutedBeamSearch(self.target_dictionary)

        if getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = PermutedSequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 1),
            pred_k=getattr(args, "predk", 8),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
        )


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []

        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))

            real_pos = sample['net_input']['pred_positions'][i] + \
                       sample['net_input']['prev_positions'][i]
            real_pos.masked_fill_(real_pos == self.tgt_dict.pad(), 9999999)
            _, sorted_indices = torch.sort(real_pos)
            real_target = sample['target'][i][sorted_indices]
            del real_pos, sorted_indices

            refs.append(decode(
                utils.strip_pad(real_target, self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])




class PermutedBeamSearch(search.Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    @torch.jit.export
    def step(self, step: int, lprobs, beam_size, scores: Optional[Tensor]):
        bsz, _, vocab_size = lprobs.size()

        # if step == 0:
        #     # at the first step all hypotheses are equally likely, so use
        #     # only the first beam
        #     lprobs = lprobs[:, ::beam_size, :].contiguous()
        # if step > 0:
        #     # make probs contain cumulative scores for each hypothesis
        #     assert scores is not None
        #     lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        top_prediction = torch.topk(
            lprobs,
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.size(-1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = torch.div(indices_buf, vocab_size)
        indices_buf.fmod_(vocab_size)
        return scores_buf, indices_buf, beams_buf



@torch.jit.script
class BeamContainer(object):
    def __init__(self, score: float, elem: Dict[str, Tensor]):
        self.score = score
        self.elem = elem

    def __lt__(self, other):
        # type: (BeamContainer) -> bool
        # Due to https://github.com/pytorch/pytorch/issues/20388,
        # this has to use old style type annotations
        # Match original behavior of sorted function when two scores are equal.
        return self.score <= other.score