import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerDecoder,\
    TransformerEncoder, base_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from .permuted_embedding import PositionalEmbeddingByPos


def make_positions(tensor, padding_idx: int, offset: int = 0, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return ((torch.cumsum(mask, dim=1).type_as(mask) + offset) * mask).long() + padding_idx


@register_model("transformer_permuted")
class TransfomerPermuted(TransformerModel):
    def __init__(self, *kargs, **kwargs):
        super(TransfomerPermuted, self).__init__(*kargs, **kwargs)

    @staticmethod
    def add_args(parser):
        super(TransfomerPermuted, TransfomerPermuted).add_args(parser)
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--decoder-learned-pred-pos', action='store_true', default=False,
                            help='use learned pred positional embeddings in the decoder')
        # fmt: on

    # @classmethod
    # def build_model(cls, args, task):
    #     """Build a new model instance."""
    #     model = super(TransfomerPermuted,
    #                   TransfomerPermuted).build_model(args, task)
    #     for m in model.modules():
    #         if isinstance(m, nn.Linear):
    #             m.reset_parameters()
    #     return model

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerPermutedEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerPermutedDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        prev_positions: Optional[Tensor] = None,
        pred_positions: Optional[Tensor] = None,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            # cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            prev_positions,
            pred_positions,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


class TransformerPermutedEncoder(TransformerEncoder):
    def forward(self, src_tokens,
                src_lengths,
                # cls_input: Optional[Tensor] = None,
                return_all_hiddens: bool = False, **kwargs):
        return super().forward(src_tokens, src_lengths, return_all_hiddens)


class TransformerPermutedDecoder(TransformerDecoder):
    def __init__(self, args, *kargs, **kwargs):
        super(TransformerPermutedDecoder, self).__init__(
            args, *kargs, **kwargs)
        self.embed_positions = PositionalEmbeddingByPos(
            args.max_target_positions,
            args.decoder_embed_dim,
            self.padding_idx,
            learned=args.decoder_learned_pos,
        )

        self.embed_pred_positions = PositionalEmbeddingByPos(
            args.max_target_positions,
            args.decoder_embed_dim,
            self.padding_idx,
            learned=args.decoder_learned_pred_pos,
            min_timescale=1e4,
            max_timescale=1e8,
            flip=True
        )

    def forward(
        self,
        prev_output_tokens,
        prev_positions: Optional[Tensor] = None,
        pred_positions: Optional[Tensor] = None,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            prev_positions,
            pred_positions,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        prev_positions: Optional[Tensor] = None,
        pred_positions: Optional[Tensor] = None,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1
        if prev_positions is None:
            prev_positions = make_positions(
                prev_output_tokens, self.padding_idx, onnx_trace=self.onnx_trace
            )
        if pred_positions is None:
            pred_positions = torch.ones_like(prev_positions)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            prev_positions = prev_positions[:, -1:]
            pred_positions = pred_positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        # embed positions
        positions = self.embed_positions(prev_positions) + \
            self.embed_pred_positions(pred_positions)

        x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("transformer_permuted", "transformer_permuted")
def transformer_permuted(args):
    args.warn_patched = getattr(args, "warn_patched", False)
    args.warn_not_patched = getattr(args, "warn_not_patched", False)
    base_architecture(args)
