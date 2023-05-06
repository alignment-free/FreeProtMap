import sys   
sys.path.append("..")
import torch
from openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from torch import nn

from esm.esmfold.v1.misc import (
    Attention,
    Dropout,
    PairToSequence,
    ResidueMLP,
    SequenceToPair,
)


class TriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        pairwise_state_dim,
        pairwise_head_width=8,
        dropout=0,
        **__kwargs,
    ):
        super().__init__()

        assert pairwise_state_dim % pairwise_head_width == 0
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width
        assert pairwise_state_dim == pairwise_num_heads * pairwise_head_width
        assert pairwise_state_dim % 2 == 0

        self.pairwise_state_dim = pairwise_state_dim



        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore
        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore

        self.mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)


        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(self, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
          mask: B x L boolean tensor of valid positions
        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2

        pairwise_state_dim = pairwise_state.shape[3]

        assert pairwise_state_dim == self.pairwise_state_dim
        # assert batch_dim == pairwise_state.shape[0]



        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out(pairwise_state, mask=tri_mask)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in(pairwise_state, mask=tri_mask)
        )
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return  pairwise_state