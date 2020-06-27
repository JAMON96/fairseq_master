# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.onnx.operators
from fairseq import utils
from torch import Tensor, nn
from fairseq.modules import SinusoidalPositionalEmbedding, LearnedPositionalEmbedding


class SinusoidalPositionalEmbeddingByPos(SinusoidalPositionalEmbedding):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024,
                 min_timescale=1, max_timescale=10000, flip=False):
        super().__init__(embedding_dim, padding_idx, init_size)
        self.max_timescale = max_timescale
        self.min_timescale = min_timescale
        self.flip = flip

    @staticmethod
    def get_embedding(positions,
                      num_embeddings: int, embedding_dim: int,
                      min_timescale: Optional[int] = 1, max_timescale: Optional[int] = 1e4,
                      flip: Optional[bool] = False
                      ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        max_timescale = float(max_timescale)
        min_timescale = float(min_timescale)
        emb = math.log(max_timescale / min_timescale) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device, dtype=torch.float)
                        * -emb) * min_timescale
        emb = positions.float().unsqueeze(-1) * emb.view(1, 1, -1)
        if flip:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, -1)], dim=-1)
        return emb

    def forward(
        self,
        positions,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        max_pos = positions.size(1)
        return SinusoidalPositionalEmbeddingByPos.get_embedding(positions,
                                                                max_pos, self.embedding_dim,
                                                                self.min_timescale, self.max_timescale, self.flip
                                                                )


class LearnedPositionalEmbeddingByPos(nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(self, positions, incremental_state=None):
        return super().forward(positions)


def PositionalEmbeddingByPos(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
        **kwargs
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbeddingByPos(
            num_embeddings, embedding_dim, padding_idx=None)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbeddingByPos(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
            **kwargs
        )
    return m
