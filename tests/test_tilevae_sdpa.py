"""Regression test for the tilevae xformers-absent SDPA attention path.

Context: on Blackwell sm_120 + Windows there is no xformers wheel with
memory_efficient_attention, so tilevae's attn2task() routes VAE attention to the
SDPA branch. That branch used to call attn_forward_new_pt2_0, which is written
for a diffusers-style Attention module (self.group_norm / self.to_q) and crashes
on sgm's vanilla AttnBlock with:

    AttributeError: 'AttnBlock' object has no attribute 'group_norm'

The fix routes the SDPA branch to attn_forward_sdpa, which mirrors AttnBlock's
own native scaled_dot_product_attention path.

This test is CPU-only (no CUDA) and asserts the tilevae task decomposition
(store_res -> pre_norm -> attn -> add_res) reproduces AttnBlock.forward() exactly.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU

import torch

from sgm.modules.diffusionmodules.model import AttnBlock
from SUPIR.utils.tilevae import attn_forward_sdpa


def test_attn_forward_sdpa_matches_native_attnblock():
    torch.manual_seed(0)
    net = AttnBlock(in_channels=32).eval()
    x = torch.randn(2, 32, 16, 16)

    with torch.no_grad():
        # Reference: the model's own forward = x + proj_out(attention(norm(x)))
        reference = net(x)

        # tilevae decomposition for the SDPA branch:
        #   store_res(x) ; pre_norm = net.norm ; attn = attn_forward_sdpa ; add_res
        res = x
        normed = net.norm(x)
        attn_out = attn_forward_sdpa(net, normed)
        result = res + attn_out

    assert result.shape == reference.shape
    assert torch.allclose(result, reference, atol=1e-5), (
        f"max abs diff {torch.max(torch.abs(result - reference)).item():.2e}"
    )


def test_old_diffusers_path_would_have_crashed():
    """Documents the original bug: the diffusers-shaped fn needs .group_norm,
    which the sgm AttnBlock does not have."""
    net = AttnBlock(in_channels=32)
    assert not hasattr(net, "group_norm")
    assert hasattr(net, "norm") and hasattr(net, "q") and hasattr(net, "proj_out")
