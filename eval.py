"""
eval.py - Benchmark evaluation for the DeepSeek-V3 project.

Supports:
  * local checkpoints produced by the DeepSeek-V3 training/fine-tuning scripts

Core benchmark families used in / closely related to the GPT-3 evaluation:
  HellaSwag, LAMBADA, PIQA, OpenBookQA, ARC-Easy, ARC-Challenge,
  WinoGrande, SuperGLUE BoolQ, SuperGLUE RTE, WSC, TriviaQA,
  StoryCloze, WebQuestions, COPA, RACE, MMLU, GSM8K, TruthfulQA, PTB,
  plus a small synthetic arithmetic evaluation.

Install dependencies in your project environment:
    pip install -U datasets tqdm tiktoken

Examples:
    python eval.py --model log/model_00250.pt
    python eval.py --model DeepSeekV3Base.pt --tasks hellaswag,lambada,piqa,mmlu,gsm8k
    python eval.py --model log/model_19072.pt --tasks all --limit 1000
"""

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        x = self.embed(idx)
        freqs_cis = self.freqs_cis[:T].to(idx.device)

        mask = None
        if not self.training and T > 1:
            mask = torch.full((T, T), float("-inf"), device=idx.device).triu_(1)

        for layer in self.layers:
            x = layer(x, 0, freqs_cis, mask)

        x = self.norm(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        else:
            loss = None

        return logits, loss


# =============================================================================
# Adapter
# =============================================================================

class Adapter:
    def __init__(self, model, config, device, name, enc):
        self.model = model.eval()
        self.config = config
        self.device = device
        self.name = name
        self.enc = enc

    @property
    def block_size(self):
        return getattr(
            self.config,
            "block_size",
            getattr(self.config, "max_seq_len", 1024),
        )

    @property
    def params(self):
        return sum(
            p.numel()
            for p in self.model.parameters()
        )

    @torch.no_grad()
    def logits(self, ids):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            )
        else:
            ids = ids.to(self.device)

        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        if ids.size(1) > self.block_size:
            ids = ids[:, -self.block_size:]

        out = self.model(ids)

        if isinstance(out, tuple):
            out = out[0]

        if hasattr(out, "logits"):
            out = out.logits

        # The checkpoint uses a padded vocabulary (50304), while the GPT-2
        # tokenizer only defines 50257 valid token IDs. Padding rows must not
        # compete during generation/evaluation.
        valid_vocab = getattr(self.enc, "n_vocab", out.size(-1))
        if out.size(-1) > valid_vocab:
            out = out.clone()
            out[..., valid_vocab:] = float("-inf")

        return out

    @torch.no_grad()
    def completion_losses(self, ids, completion_start):
        """Return (sum NLL, mean NLL) over completion tokens only."""
        full_len = len(ids)

        if full_len < 2:
            return float("inf"), float("inf")

        if full_len > self.block_size:
            trim = full_len - self.block_size
            ids = ids[-self.block_size:]
            completion_start = max(
                1,
                completion_start - trim
            )

        x = ids[:-1]
        y = torch.tensor(
            ids[1:],
            dtype=torch.long,
            device=self.device,
        )

        logits = self.logits(x)

        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y,
            reduction="none",
        )

        first = max(
            0,
            completion_start - 1
        )
        losses = losses[first:]

        if losses.numel() == 0:
            return float("inf"), float("inf")

        return (
            losses.sum().item(),
            losses.mean().item(),
        )

    @torch.no_grad()
    def greedy_generate(self, prompt, max_new_tokens=20):
        ids = self.enc.encode(prompt)
        out = list(ids)

        for _ in range(max_new_tokens):
            logits = self.logits(out)
            nxt = int(
                torch.argmax(
                    logits[:, -1, :],
                    dim=-1
                ).item()
            )
            out.append(nxt)

            if nxt == self.enc.eot_token:
                break

        return self.enc.decode(
            out[len(ids):]
        ).strip()


def load_model(identifier, device, enc):
    p = Path(identifier)

    if not p.exists() or not p.is_file():
        raise FileNotFoundError(
            f"DeepSeek-V3 evaluation expects a local "
            f"project checkpoint, but '{identifier}' "
            f"was not found."
        )

    # Load checkpoint on CPU first. This avoids putting a second copy of the
    # checkpoint on GPU while the model object is being constructed.
    ckpt = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"{p} is not a valid DeepSeek-V3 project checkpoint"
        )

    cfg = DeepSeekV3Config(**ckpt["config"])
    model = DeepSeekV3(cfg, init_weights=False)

    missing, unexpected = model.load_state_dict(
        ckpt["model"],
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {p}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.to(device).eval()

    return Adapter(
        model,
        cfg,
        device,
        str(p),
        enc,
    )


# =============================================================================
# Dataset loading / helpers
# =============================================================================

def ds(path, name=None, split="validation"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install/update datasets: pip install -U datasets"
        ) from exc

    kwargs = {
        "path": path,
        "split": split,
    }

    if name:
        kwargs["name"] = name

    try:
        return load_dataset(**kwargs)
    except Exception as exc:
        config_text = f" / config '{name}'" if name else ""
        raise RuntimeError(
            f"Could not load dataset '{path}'{config_text} "
            f"/ split '{split}'. Error: {exc}"
        ) from exc


def limited(dataset, limit):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def mc_score(model, context, completion, normalized=True):
    c = model.enc.encode(context)
    a = model.enc.encode(completion)
    if not a:
        return float("inf")
    return model.completion_losses(c + a, len(c))[1 if normalized else 0]


def normalize_answer(s):
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


# =============================================================================
# Existing Evaluation Functions
# =============================================================================

def eval_hellaswag(model, limit):
    data = limited(ds("allenai/hellaswag", split="validation"), limit)
    correct_sum = correct_norm = total = 0
    for ex in tqdm(data, desc="HellaSwag"):
        losses_sum, losses_norm = [], []
        for ending in ex["endings"]:
            c = model.enc.encode(ex["ctx"])
            a = model.enc.encode(" " + ending)
            s, m = model.completion_losses(c + a, len(c))
            losses_sum.append(s)
            losses_norm.append(m)
        pred_s = min(range(4), key=losses_sum.__getitem__)
        pred_n = min(range(4), key=losses_norm.__getitem__)
        label = int(ex["label"])
        correct_sum += pred_s == label
        correct_norm += pred_n == label
        total += 1
    return {
        "accuracy": correct_sum / total if total else float("nan"),
        "accuracy_norm": correct_norm / total if total else float("nan"),
        "total": total,
    }


def eval_lambada(model, limit):
    data = ds("EleutherAI/lambada_openai", "default", "test")
    data = limited(data, limit)
    correct = total = target_nll = target_tokens = 0
    for ex in tqdm(data, desc="LAMBADA"):
        text = ex.get("text") or ex.get("sentence") or ex.get("context")
        if not text:
            continue
        words = text.rstrip().split()
        if len(words) < 2:
            continue
        context, target = " ".join(words[:-1]), words[-1]
        c = model.enc.encode(context)
        a = model.enc.encode(" " + target)
        s, _ = model.completion_losses(c + a, len(c))
        generated = list(c)
        pred = []
        for _ in a:
            logits = model.logits(generated)
            nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            pred.append(nxt)
            generated.append(nxt)
        correct += pred == a
        total += 1
        target_nll += s
        target_tokens += len(a)
    return {
        "accuracy": correct / total if total else float("nan"),
        "perplexity": math.exp(target_nll / target_tokens) if target_tokens else float("nan"),
        "total": total,
    }


def eval_piqa(model, limit):
    data = limited(ds("lighteval/piqa", split="validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="PIQA"):
        losses = [mc_score(model, ex["goal"] + "\nAnswer:", " " + ex[k]) for k in ("sol1", "sol2")]
        correct += min(range(2), key=losses.__getitem__) == int(ex["label"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_openbookqa(model, limit):
    data = limited(ds("allenai/openbookqa", "main", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="OpenBookQA"):
        texts = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        losses = [mc_score(model, ex["question_stem"] + "\nAnswer:", " " + t) for t in texts]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = labels.index(ex["answerKey"])
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_arc(model, limit, challenge):
    cfg = "ARC-Challenge" if challenge else "ARC-Easy"
    data = limited(ds("allenai/ai2_arc", cfg, "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc=cfg):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        losses = [mc_score(model, ex["question"] + "\nAnswer:", " " + t) for t in texts]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = labels.index(ex["answerKey"])
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_winogrande(model, limit):
    data = limited(ds("allenai/winogrande", "winogrande_xl", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="WinoGrande"):
        prefix, suffix = ex["sentence"].split("_", 1)
        losses = []
        for key in ("option1", "option2"):
            c = prefix
            a = ex[key] + suffix
            losses.append(mc_score(model, c, a))
        pred = min(range(2), key=losses.__getitem__)
        correct += pred == int(ex["answer"]) - 1
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_boolq(model, limit):
    data = limited(ds("google/boolq", split="validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="BoolQ"):
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        scores = [mc_score(model, prompt, x) for x in (" yes", " no")]
        pred = scores[0] < scores[1]
        correct += pred == bool(ex["answer"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_rte(model, limit):
    data = limited(ds("nyu-mll/glue", "rte", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="RTE"):
        prompt = f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}\nAnswer:"
        scores = [mc_score(model, prompt, " entailment"), mc_score(model, prompt, " not entailment")]
        pred = min(range(2), key=scores.__getitem__)
        correct += pred == int(ex["label"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_wsc(model, limit):
    data = limited(ds("aps/super_glue", "wsc.fixed", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="WSC"):
        prompt = f"Text: {ex['text']}\nQuestion: Does \"{ex['span1_text']}\" refer to \"{ex['span2_text']}\"?\nAnswer:"
        scores = [mc_score(model, prompt, " No"), mc_score(model, prompt, " Yes")]
        pred = 1 if scores[1] < scores[0] else 0
        correct += pred == int(ex["label"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_triviaqa(model, limit):
    data = limited(ds("mandarjoshi/trivia_qa", "rc.nocontext", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="TriviaQA"):
        # Limita la respuesta a la primera línea generada
        pred = model.greedy_generate(f"Question: {ex['question']}\nAnswer:", 16).strip().split('\n')[0]
        refs = list(ex["answer"]["aliases"]) + [ex["answer"]["value"]]
        
        pred_norm = normalize_answer(pred)
        # Evaluamos por inclusión (substring) y no por coincidencia exacta
        ok = any(normalize_answer(r) in pred_norm for r in refs if r)
        correct += ok
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_arithmetic(model, limit, seed=42):
    rng = random.Random(seed)
    n = limit or 500
    correct = 0
    for _ in tqdm(range(n), desc="Arithmetic"):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(("+", "*"))
        answer = a + b if op == "+" else a * b
        prompt = f"{a} {op} {b} ="
        generated = model.greedy_generate(
            prompt,
            max_new_tokens=max(8, len(str(answer)) + 2),
        ).strip()
        pred = generated.split()[0] if generated else ""
        correct += pred == str(answer)
    return {"accuracy": correct / n, "total": n}


def eval_storycloze(model, limit):
    data = limited(
        ds("MoE-UNC/story_cloze", split="validation"),
        limit
    )
    correct = total = 0
    for ex in tqdm(data, desc="StoryCloze"):
        ctx = (
            f"{ex['input_sentence_1']} "
            f"{ex['input_sentence_2']} "
            f"{ex['input_sentence_3']} "
            f"{ex['input_sentence_4']}"
        )

        losses = [
            mc_score(model,ctx," " + ex["sentence_quiz1"]),
            mc_score(model,ctx," " + ex["sentence_quiz2"]),
        ]

        pred = min(range(2), key=losses.__getitem__)
        gold = int(ex["answer_right_ending"]) - 1

        correct += pred == gold
        total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
    }


def eval_webquestions(model, limit):
    data = limited(ds("stanfordnlp/web_questions", split="test"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="WebQuestions"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        pred = model.greedy_generate(prompt, max_new_tokens=16).strip().split('\n')[0]
        refs = ex.get("answers", [])
        
        pred_norm = normalize_answer(pred)
        # Evaluamos por inclusión (substring)
        ok = any(normalize_answer(r) in pred_norm for r in refs if r)
        correct += int(ok)
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_copa(model, limit):
    data = limited(ds("aps/super_glue", "copa", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="COPA"):
        connector = " because" if ex["question"] == "cause" else " so"
        prompt = ex["premise"] + connector
        losses = [
            mc_score(model, prompt, " " + ex["choice1"]),
            mc_score(model, prompt, " " + ex["choice2"]),
        ]
        pred = min(range(2), key=losses.__getitem__)
        correct += pred == int(ex["label"])
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_race(model, limit):
    # Cambiado a 'ehovy/race'
    data = limited(ds("ehovy/race", "all", "validation"), limit)
    correct = total = 0
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    for ex in tqdm(data, desc="RACE"):
        prompt = f"Article: {ex['article']}\nQuestion: {ex['question']}\nAnswer:"
        losses = [mc_score(model, prompt, " " + opt) for opt in ex["options"]]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = mapping.get(str(ex["answer"]).strip().upper(), 0)
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_mmlu(model, limit):
    data = limited(ds("cais/mmlu", "all", "test"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="MMLU"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        losses = [mc_score(model, prompt, " " + str(opt)) for opt in ex["choices"]]
        pred = min(range(len(losses)), key=losses.__getitem__)
        correct += pred == int(ex["answer"])
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_gsm8k(model, limit):
    # Cambiado a 'openai/gsm8k'
    data = limited(ds("openai/gsm8k", "main", "test"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="GSM8K"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        generated = model.greedy_generate(prompt, max_new_tokens=64)
        target_str = ex["answer"].split("####")[-1].strip().replace(",", "")
        gen_numbers = re.findall(r"-?\d+(?:\.\d+)?", generated.replace(",", ""))
        pred = gen_numbers[-1] if gen_numbers else ""
        correct += pred == target_str
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_truthfulqa(model, limit):
    data = limited(ds("truthfulqa/truthful_qa", "multiple_choice", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="TruthfulQA"):
        prompt = f"Q: {ex['question']}\nA:"
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        losses = [mc_score(model, prompt, " " + choice) for choice in choices]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = labels.index(1) if 1 in labels else 0
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


# =============================================================================
# Registry / CLI
# =============================================================================

TASKS = {
    "hellaswag": eval_hellaswag,
    "lambada": eval_lambada,
    "piqa": eval_piqa,
    "openbookqa": eval_openbookqa,
    "arc_easy": lambda m, l: eval_arc(m, l, False),
    "arc_challenge": lambda m, l: eval_arc(m, l, True),
    "winogrande": eval_winogrande,
    "boolq": eval_boolq,
    "rte": eval_rte,
    "wsc": eval_wsc,
    "triviaqa": eval_triviaqa,
    "arithmetic": eval_arithmetic,
    "storycloze": eval_storycloze,
    "webquestions": eval_webquestions,
    "copa": eval_copa,
    "race": eval_race,
    "mmlu": eval_mmlu,
    "gsm8k": eval_gsm8k,
    "truthfulqa": eval_truthfulqa,
}

DEFAULT_TASKS = [
    "hellaswag",
    "lambada",
    "piqa",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "wsc",
    "boolq",
    "rte",
    "triviaqa",
    "storycloze",
    "webquestions",
    "copa",
    "race",
    "mmlu",
    "gsm8k",
    "truthfulqa"
]

def validate_datasets(task_names):
    """Check that all selected benchmark datasets can be loaded."""
    checks = {
    "hellaswag": ("allenai/hellaswag", None, "validation"),
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "piqa": ("lighteval/piqa", None, "validation"),
    "openbookqa": ("allenai/openbookqa", "main", "validation"),
    "arc_easy": ("allenai/ai2_arc","ARC-Easy","validation"),
    "arc_challenge": ("allenai/ai2_arc","ARC-Challenge","validation"),
    "winogrande": ("allenai/winogrande","winogrande_xl","validation"),
    "wsc": ("aps/super_glue","wsc.fixed","validation"),
    "boolq": ("google/boolq",None,"validation"),
    "rte": ("nyu-mll/glue","rte","validation"),
    "triviaqa": ("mandarjoshi/trivia_qa","rc.nocontext","validation"),
    "storycloze": ("MoE-UNC/story_cloze",None,"validation"),
    "webquestions": ("stanfordnlp/web_questions",None,"test"),
    "copa": ("aps/super_glue","copa","validation"),
    "race": ("ehovy/race","all","validation"),
    "mmlu": ("cais/mmlu","all","test"),
    "gsm8k": ("openai/gsm8k","main","test"),
    "truthfulqa": ("truthfulqa/truthful_qa","multiple_choice","validation"),
    }

    for task in task_names:
        if task == "arithmetic" or task not in checks:
            continue

        path, name, split = checks[task]
        label = f"{path}" + (f" [{name}]" if name else "")
        print(f"Checking {task}: {label}")

        data = ds(path, name, split)

        if len(data) == 0:
            raise RuntimeError(f"{task} loaded but contains zero examples.")

        print(f"  OK - {len(data):,} examples")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="Path(s) to local DeepSeek-V3 project checkpoint(s)",
    )
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated tasks or 'all'")
    p.add_argument("--limit", type=int, default=None, help="Maximum examples per task")
    p.add_argument("--device", default=None, help="cuda / cpu; auto if omitted")
    p.add_argument("--output", default="eval_results.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--check-datasets",
        action="store_true",
        help="Check/download selected datasets and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    enc = tiktoken.get_encoding("gpt2")
    task_names = list(TASKS) if args.tasks.lower() == "all" else [x.strip() for x in args.tasks.split(",") if x.strip()]

    unknown = [x for x in task_names if x not in TASKS]
    if unknown:
        raise ValueError(
            f"Unknown tasks {unknown}. Available: {', '.join(TASKS)}"
        )

    if args.check_datasets:
        validate_datasets(task_names)
        print("\nAll selected datasets are accessible.")
        return

    results = {
        "meta": {
            "device": device,
            "tasks": task_names,
            "limit": args.limit,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "models": {},
    }

    for identifier in args.model:
        print("\n" + "=" * 80)
        print(f"MODEL: {identifier}")
        print("=" * 80)
        model = load_model(identifier, device, enc)
        info = {
            "parameters": model.params,
            "parameters_millions": model.params / 1e6,
            "n_layer": getattr(model.config, "n_layer", None),
            "n_head": getattr(model.config, "n_head", None),
            "n_embd": getattr(model.config, "n_embd", None),
            "tasks": {},
        }
        print(f"Parameters: {model.params:,} ({model.params / 1e6:.2f}M)")
        for task in task_names:
            print(f"\n--- {task} ---")
            t0 = time.time()
            try:
                if task == "arithmetic":
                    r = eval_arithmetic(model, args.limit, args.seed)
                else:
                    r = TASKS[task](model, args.limit)
                r["seconds"] = time.time() - t0
                info["tasks"][task] = r
                print(json.dumps(r, indent=2))
            except Exception as exc:
                print(f"ERROR: {exc}")
                info["tasks"][task] = {"error": str(exc), "seconds": time.time() - t0}
        results["models"][model.name] = info
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Results saved to: {out}")
    print("=" * 80)
    for name, info in results["models"].items():
        print(f"\n{name} ({info['parameters_millions']:.2f}M)")
        for task, r in info["tasks"].items():
            if "error" in r:
                print(f"  {task:16s} ERROR")
            else:
                bits = []
                if "accuracy" in r:
                    bits.append(f"acc={r['accuracy']:.4f}")
                if "accuracy_norm" in r:
                    bits.append(f"acc_norm={r['accuracy_norm']:.4f}")
                if "perplexity" in r:
                    bits.append(f"ppl={r['perplexity']:.4f}")
                print(f"  {task:16s} " + ", ".join(bits))


if __name__ == "__main__":
    main()