"""
Fine-tune a GPT-2-style model on the English portion of OpenAssistant/oasst1.

This script is self-contained:
  1. Downloads/caches OpenAssistant/oasst1 through Hugging Face Datasets.
  2. Keeps English messages only.
  3. Reconstructs root -> assistant-response conversation examples.
  4. Tokenizes them with the GPT-2 tokenizer.
  5. Writes fixed-length training/validation shards under data/oasst1_shards/.
  6. Fine-tunes a local DeepSeekV3 checkpoint, computing loss ONLY on assistant tokens.

The generated checkpoint is compatible with the project's generate.py/eval.py:
    {
        "model": state_dict,
        "config": {"block_size", "vocab_size", "n_layer", "n_head", "n_embd"},
        "step": ...,
        "val_loss": ...,
    }

Recommended first run:
    python finetune_oasst.py --checkpoint DeepSeekV3Base.pt --steps 2000

Resume:
    python finetune_oasst.py --checkpoint log/oasst/model_2000.pt --resume

Rebuild only the dataset:
    python finetune_oasst.py --build-data-only

Dependencies:
    pip install torch tiktoken datasets tqdm numpy
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# =============================================================================
# DeepSeek-V3 Architecture & Adapter
# =============================================================================
#
# This evaluation file uses the same model naming/configuration convention as
# the DeepSeekV3 fine-tuning script:
#
#     DeepSeekV3Config
#     DeepSeekV3
#
# The checkpoint must therefore contain:
#     {
#         "model": state_dict,
#         "config": vars(DeepSeekV3Config),
#         ...
#     }
#
# MLA is evaluated through a full-sequence path (no KV-cache writes) because
# benchmark completion scoring needs logits for every token. This is the same
# mathematical MLA used by the fine-tuning model, without inference caching.
# =============================================================================

from typing import Tuple, Optional, Literal

try:
    from kernel import act_quant, weight_dequant, fp8_gemm
except ImportError:
    act_quant = None
    weight_dequant = None
    fp8_gemm = None

import torch.distributed as dist


world_size = 1
rank = 0
kernel_block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"


@dataclass
class DeepSeekV3Config:
    # Common GPT-2 training fields
    block_size: int = 1024
    vocab_size: int = 50304  # padded GPT-2 tokenizer vocabulary
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512

    # DeepSeek-V3-inspired fields
    max_batch_size: int = 8
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None

    # Dense/MoE FFN
    inter_dim: int = 1536
    moe_inter_dim: int = 224
    n_dense_layers: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 2
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # RoPE / YaRN-compatible settings
    max_seq_len: int = 1024
    original_seq_len: int = 1024
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale, kernel_block_size)
        return F.linear(x, weight, bias)
    else:
        if act_quant is None or fp8_gemm is None:
            raise RuntimeError("FP8 GEMM requested but kernel.py is unavailable")
        x, scale = act_quant(x, kernel_block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.float32
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + kernel_block_size - 1) // kernel_block_size
            scale_in_features = (in_features + kernel_block_size - 1) // kernel_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: DeepSeekV3Config) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepSeekV3Config): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: DeepSeekV3Config):
        super().__init__()
        self.dim = args.n_embd
        self.n_heads = args.n_head
        self.n_local_heads = args.n_head // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.wo.NANOGPT_SCALE_INIT = 1
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Full-sequence MLA path used by evaluation.

        This implementation intentionally has no KV cache. Benchmark scoring
        needs logits for every token in the supplied sequence, so keeping a
        decode-time cache here would add state and memory without helping the
        benchmark workload.
        """
        del start_pos  # kept for API compatibility with Block

        bsz, seqlen, _ = x.size()

        if freqs_cis is None:
            raise ValueError("freqs_cis is required for MLA evaluation")

        # ---------------------------------------------------------
        # Query projection
        # ---------------------------------------------------------
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(
                self.q_norm(
                    self.wq_a(x)
                )
            )

        q = q.view(
            bsz,
            seqlen,
            self.n_local_heads,
            self.qk_head_dim,
        )

        q_nope, q_pe = torch.split(
            q,
            [
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # ---------------------------------------------------------
        # KV latent projection
        # ---------------------------------------------------------
        kv = self.wkv_a(x)

        kv, k_pe = torch.split(
            kv,
            [
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2),
            freqs_cis,
        )

        # ---------------------------------------------------------
        # Absorb WKV-B into Q / V path.
        # This is the same mathematical "absorb" formulation used by
        # the adapted DeepSeek implementation, but without cache writes.
        # ---------------------------------------------------------
        wkv_b = self.wkv_b.weight

        if getattr(self.wkv_b, "scale", None) is not None:
            if weight_dequant is None:
                raise RuntimeError(
                    "Quantized MLA weight requires the project's kernel.py"
                )

            wkv_b = weight_dequant(
                self.wkv_b.weight,
                self.wkv_b.scale,
                kernel_block_size,
            )

        wkv_b = wkv_b.view(
            self.n_local_heads,
            -1,
            self.kv_lora_rank,
        )

        q_nope = torch.einsum(
            "bshd,hdc->bshc",
            q_nope,
            wkv_b[:, :self.qk_nope_head_dim],
        )

        latent_kv = self.kv_norm(kv)

        # ---------------------------------------------------------
        # Attention scores
        # ---------------------------------------------------------
        scores = (
            torch.einsum(
                "bshc,btc->bsht",
                q_nope,
                latent_kv,
            )
            +
            torch.einsum(
                "bshr,btr->bsht",
                q_pe,
                k_pe.squeeze(2),
            )
        ) * self.softmax_scale

        if mask is not None:
            scores = scores + mask.unsqueeze(1)

        scores = F.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).type_as(x)

        # ---------------------------------------------------------
        # Values
        # ---------------------------------------------------------
        values = torch.einsum(
            "bsht,btc->bshc",
            scores,
            latent_kv,
        )

        values = torch.einsum(
            "bshc,hdc->bshd",
            values,
            wkv_b[:, -self.v_head_dim:],
        )

        # ---------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------
        return self.wo(values.flatten(2))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w2.NANOGPT_SCALE_INIT = 1
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: DeepSeekV3Config):
        """
        Initializes the Gate module.

        Args:
            args (DeepSeekV3Config): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.n_embd
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: DeepSeekV3Config):
        """
        Initializes the MoE module.

        Args:
            args (DeepSeekV3Config): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.n_embd
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.n_embd, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.n_embd, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: DeepSeekV3Config):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (DeepSeekV3Config): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.n_embd, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.n_embd)
        self.ffn_norm = RMSNorm(args.n_embd)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DeepSeekV3(nn.Module):
    """
    DeepSeek-V3-inspired decoder-only model with GPT-2-style training API.

    The architecture keeps MLA + shared/routed MoE from the DeepSeek-style
    implementation, while exposing the same forward/checkpoint/optimizer
    conventions as the GPT-2 training file used in this project.
    """

    def __init__(
        self,
        config: DeepSeekV3Config,
        init_weights: bool = True,
    ):
        super().__init__()
        self.config = config
        self.init_weights = init_weights

        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Keep parameters in FP32, like the DeepSeekV3 training script. BF16 is
        # used through autocast during the forward/backward pass.
        Linear.dtype = torch.float32
        Linear.scale_fmt = config.scale_fmt

        self.max_seq_len = config.block_size
        self.embed = ParallelEmbedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [Block(layer_id, config) for layer_id in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd)
        self.head = ColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            dtype=torch.float32,
        )
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config),
            persistent=False,
        )

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        x = self.embed(idx)
        freqs_cis = self.freqs_cis[:T].to(idx.device)

        attn_mask = None
        if not self.training and T > 1:
            attn_mask = torch.full((T, T), float("-inf"), device=idx.device).triu_(1)

        for layer in self.layers:
            x = layer(x, 0, freqs_cis, attn_mask)

        x = self.norm(x)
        logits = self.head(x)

        if targets is not None:
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            if loss_mask is not None:
                loss_mask = loss_mask.reshape(-1).to(per_token_loss.dtype)
                denom = loss_mask.sum().clamp_min(1.0)
                loss = (per_token_loss * loss_mask).sum() / denom
            else:
                loss = per_token_loss.mean()
        else:
            loss = None

        return logits, loss


# -----------------------------------------------------------------------------
# OASST1 -> fixed-length shards
# -----------------------------------------------------------------------------

SPECIAL_USER = "<|user|>\n"
SPECIAL_ASSISTANT = "<|assistant|>\n"


def load_oasst_split(split):
    """Load the current Parquet-backed OASST1 split."""
    # OASST1 is currently Parquet-backed on HF and exposes train/validation.
    return load_dataset("OpenAssistant/oasst1", split=split)


def build_examples(dataset, max_examples=None):
    """Reconstruct English root-to-assistant-turn examples from flat messages.

    For every assistant message, its complete ancestor chain is used as context.
    This gives the model both single-turn and multi-turn SFT examples without
    needing the separate .trees.jsonl.gz files.
    """
    rows = {}
    children = {}

    for row in dataset:
        if row.get("lang") != "en":
            continue
        if row.get("deleted", False):
            continue
        mid = row["message_id"]
        rows[mid] = row
        parent = row.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(mid)

    # Cache reconstructed chains.
    chain_cache = {}

    def chain(mid):
        if mid in chain_cache:
            return chain_cache[mid]
        row = rows[mid]
        parent = row.get("parent_id")
        if parent and parent in rows:
            result = chain(parent) + [mid]
        else:
            result = [mid]
        chain_cache[mid] = result
        return result

    examples = []
    for mid, row in rows.items():
        if row.get("role") != "assistant":
            continue
        ids = chain(mid)
        messages = [rows[x] for x in ids]
        # A valid SFT example should start with a prompter and alternate roles.
        if not messages or messages[0].get("role") != "prompter":
            continue
        valid = True
        for i, msg in enumerate(messages):
            expected = "prompter" if i % 2 == 0 else "assistant"
            if msg.get("role") != expected:
                valid = False
                break
            if not isinstance(msg.get("text"), str) or not msg["text"].strip():
                valid = False
                break
        if not valid:
            continue

        examples.append(messages)
        if max_examples and len(examples) >= max_examples:
            break

    return examples


def encode_conversation(messages, enc, block_size):
    """Encode one conversation ending in an assistant message.

    Returns fixed-size token and mask arrays. Prompt tokens have mask=0;
    assistant tokens + EOS have mask=1.
    """
    token_ids = []
    loss_mask = []
    eot = enc.eot_token

    for msg in messages:
        if msg["role"] == "prompter":
            prefix = enc.encode(SPECIAL_USER)
            text_tokens = enc.encode(msg["text"].strip() + "\n")
            token_ids.extend(prefix + text_tokens)
            loss_mask.extend([0] * (len(prefix) + len(text_tokens)))
        else:
            prefix = enc.encode(SPECIAL_ASSISTANT)
            text_tokens = enc.encode(msg["text"].strip(), disallowed_special=())
            part = prefix + text_tokens + [eot]
            token_ids.extend(part)
            loss_mask.extend([0] * len(prefix) + [1] * (len(text_tokens) + 1))

    # Keep the target answer. If the complete conversation is too long, trim
    # from the left; this preserves the final assistant response as much as
    # possible, which is what the loss is training on.
    if len(token_ids) > block_size:
        token_ids = token_ids[-block_size:]
        loss_mask = loss_mask[-block_size:]

    # We need a next-token target, so pad to block_size+1.
    if len(token_ids) < 2:
        return None

    # For fixed-shape training, create idx of block_size and targets of block_size.
    # If shorter than block_size, pad with EOS and mask=0.
    idx = token_ids[:-1]
    targets = token_ids[1:]
    mask = loss_mask[1:]

    pad_len = block_size - len(idx)
    if pad_len > 0:
        idx += [eot] * pad_len
        targets += [eot] * pad_len
        mask += [0] * pad_len

    return (
        np.asarray(idx, dtype=np.uint16),
        np.asarray(targets, dtype=np.uint16),
        np.asarray(mask, dtype=np.uint8),
    )


def write_shards(examples, enc, block_size, out_dir, prefix, shard_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale shards of this split so rebuilding cannot silently mix data.
    for p in out_dir.glob(f"{prefix}-*.npz"):
        p.unlink()

    shard_idx = 0
    count = 0
    buffers = [[], [], []]

    def flush():
        nonlocal shard_idx, count
        if not buffers[0]:
            return
        tokens = np.stack(buffers[0])
        targets = np.stack(buffers[1])
        masks = np.stack(buffers[2])
        path = out_dir / f"{prefix}-{shard_idx:05d}.npz"
        np.savez(path, tokens=tokens, targets=targets, masks=masks)
        count += len(tokens)
        shard_idx += 1
        buffers[0].clear(); buffers[1].clear(); buffers[2].clear()

    for messages in tqdm(examples, desc=f"Tokenizing {prefix}"):
        item = encode_conversation(messages, enc, block_size)
        if item is None:
            continue
        for i, arr in enumerate(item):
            buffers[i].append(arr)
        if len(buffers[0]) >= shard_size:
            flush()

    flush()
    return count, shard_idx


def prepare_data(args, enc):
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "metadata.json"

    if marker.exists() and not args.rebuild_data:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("block_size") == args.block_size:
            print(f"Using existing OASST shards in {out_dir}")
            return meta

    print("Downloading/loading OpenAssistant/oasst1...")
    train_ds = load_oasst_split("train")
    val_ds = load_oasst_split("validation")

    print(f"Raw train: {len(train_ds):,} messages")
    print(f"Raw validation: {len(val_ds):,} messages")

    train_examples = build_examples(train_ds, args.max_train_examples)
    val_examples = build_examples(val_ds, args.max_val_examples)

    print(f"English reconstructed train examples: {len(train_examples):,}")
    print(f"English reconstructed validation examples: {len(val_examples):,}")

    train_count, train_shards = write_shards(
        train_examples, enc, args.block_size, out_dir, "train", args.shard_size
    )
    val_count, val_shards = write_shards(
        val_examples, enc, args.block_size, out_dir, "val", args.shard_size
    )

    meta = {
        "dataset": "OpenAssistant/oasst1",
        "language": "en",
        "block_size": args.block_size,
        "train_examples": train_count,
        "val_examples": val_count,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "format": "npz(tokens, targets, masks)",
    }
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return meta


# -----------------------------------------------------------------------------
# Sharded batch loader
# -----------------------------------------------------------------------------

class ShardLoader:
    def __init__(self, data_dir, split, batch_size, device, seed=1337):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.shards = sorted(self.data_dir.glob(f"{split}-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No {split} shards in {self.data_dir}")
        self.arrays = [np.load(p, mmap_mode="r") for p in self.shards]
        self.lengths = [len(x["tokens"]) for x in self.arrays]
        self.total = sum(self.lengths)
        self.cumulative = np.cumsum(self.lengths)
        print(f"{split}: {len(self.shards)} shards, {self.total:,} examples")

    def _locate(self, indices):
        shard_ids = np.searchsorted(self.cumulative, indices, side="right")
        prev = np.concatenate(([0], self.cumulative[:-1]))
        local = indices - prev[shard_ids]
        return shard_ids, local

    def get_batch(self):
        indices = self.rng.integers(0, self.total, size=self.batch_size)
        shard_ids, local_ids = self._locate(indices)
        tokens = np.empty((self.batch_size, self.arrays[0]["tokens"].shape[1]), dtype=np.uint16)
        targets = np.empty_like(tokens)
        masks = np.empty((self.batch_size, tokens.shape[1]), dtype=np.uint8)
        for sid in np.unique(shard_ids):
            sel = np.where(shard_ids == sid)[0]
            li = local_ids[sel]
            tokens[sel] = self.arrays[sid]["tokens"][li]
            targets[sel] = self.arrays[sid]["targets"][li]
            masks[sel] = self.arrays[sid]["masks"][li]
        x = torch.from_numpy(tokens.astype(np.int64)).to(self.device)
        y = torch.from_numpy(targets.astype(np.int64)).to(self.device)
        m = torch.from_numpy(masks.astype(np.float32)).to(self.device)
        return x, y, m


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError("Checkpoint must contain 'model' and 'config'.")
    config = DeepSeekV3Config(**ckpt["config"])
    model = DeepSeekV3(config)
    model.load_state_dict(ckpt["model"])
    return model, config, ckpt


def get_lr(step, warmup, total_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if total_steps <= warmup:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model, loader, steps, autocast_dtype):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y, m = loader.get_batch()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(x.device.type == "cuda")):
                _, loss = model(x, y, m)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, config, optimizer, step, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": vars(config),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = tiktoken.get_encoding("gpt2")

    # Build data before loading the model so --build-data-only needs no checkpoint.
    prepare_data(args, enc)
    if args.build_data_only:
        return

    model, config, original_ckpt = load_checkpoint(args.checkpoint, device)
    if config.block_size != args.block_size:
        raise ValueError(
            f"Checkpoint block_size={config.block_size}, but shard block_size={args.block_size}. "
            "Use --block-size matching the checkpoint."
        )

    model = model.to(device)
    
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume and original_ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(original_ckpt["optimizer"])
        start_step = int(original_ckpt.get("step", 0))
        print(f"Resuming optimizer at step {start_step}")

    train_loader = ShardLoader(args.data_dir, "train", args.batch_size, device, args.seed)
    val_loader = ShardLoader(args.data_dir, "val", args.batch_size, device, args.seed + 1)

    use_cuda = device.type == "cuda"
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and not bf16))

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"AMP: {'bf16' if bf16 else ('fp16' if use_cuda else 'off')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            x, y, m = train_loader.get_batch()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_cuda):
                _, loss = model(x, y, m)
                loss = loss / args.grad_accum
            accum_loss += loss.item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        real_step = step + 1
        if real_step % args.log_interval == 0 or real_step == 1:
            elapsed = time.time() - t0
            print(
                f"step {real_step:6d}/{args.steps} | "
                f"lr {lr:.2e} | train_loss {accum_loss:.4f} | "
                f"tok/s ~{args.batch_size * args.grad_accum * config.block_size / max(elapsed, 1e-6):.0f}"
            )
            t0 = time.time()

        if real_step % args.eval_interval == 0 or real_step == args.steps:
            val_loss = evaluate(model, val_loader, args.eval_steps, autocast_dtype)
            print(f"validation loss: {val_loss:.4f}")

            # compiled models expose the original module through _orig_mod
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = out_dir / f"model_{real_step}.pt"
            save_checkpoint(save_model, config, optimizer, real_step, val_loss, ckpt_path)
            print(f"saved: {ckpt_path}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = out_dir / "model_best.pt"
                save_checkpoint(save_model, config, optimizer, real_step, val_loss, best_path)
                print(f"saved best: {best_path}")

    print("Fine-tuning complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="DeepSeekV3Base.pt")
    p.add_argument("--data-dir", default="data/oasst1_shards")
    p.add_argument("--out-dir", default="log/oasst")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--rebuild-data", action="store_true")
    p.add_argument("--build-data-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
