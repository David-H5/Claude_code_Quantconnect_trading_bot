"""
Multi-Head Attention Layer for PPO Enhancement (UPGRADE-010 Sprint 2)

Provides attention mechanisms for multi-asset portfolio optimization.
Designed to integrate with existing PPO optimizer in llm/ppo_weight_optimizer.py.

Features:
- Multi-head self-attention
- Asset correlation encoding
- Position encoding for temporal features
- Scaled dot-product attention
- Compatible with existing PPO infrastructure

QuantConnect Compatible: Yes (no external dependencies required)

Usage:
    from models.attention_layer import (
        MultiHeadAttention,
        AttentionConfig,
        create_attention_layer,
    )

    # Create attention layer
    attention = create_attention_layer(
        embed_dim=64,
        num_heads=4,
    )

    # Forward pass
    output, weights = attention.forward(
        query=asset_features,  # Shape: (batch, num_assets, embed_dim)
        key=asset_features,
        value=asset_features,
    )

    # Get asset correlations from attention weights
    correlations = attention.get_correlation_matrix(weights)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class AttentionType(Enum):
    """Type of attention mechanism."""

    SELF = "self"  # Self-attention
    CROSS = "cross"  # Cross-attention
    MASKED = "masked"  # Masked attention (causal)


@dataclass
class AttentionConfig:
    """Configuration for attention layer."""

    embed_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    use_bias: bool = True
    attention_type: AttentionType = AttentionType.SELF


@dataclass
class AttentionWeights:
    """Attention weight outputs."""

    weights: list[list[list[float]]]  # (batch, query_len, key_len)
    head_weights: list[list[list[list[float]]]] | None = None  # Per-head weights


@dataclass
class AttentionOutput:
    """Output from attention forward pass."""

    output: list[list[list[float]]]  # (batch, seq_len, embed_dim)
    attention_weights: AttentionWeights
    correlation_matrix: list[list[float]] | None = None


def scaled_dot_product_attention(
    query: list[list[float]],
    key: list[list[float]],
    value: list[list[float]],
    mask: list[list[float]] | None = None,
    dropout: float = 0.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: Query vectors (seq_len, head_dim)
        key: Key vectors (seq_len, head_dim)
        value: Value vectors (seq_len, head_dim)
        mask: Optional attention mask
        dropout: Dropout rate (not applied in this implementation)

    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = len(query[0]) if query and query[0] else 1
    scale = math.sqrt(d_k)

    # Compute QK^T
    seq_len = len(query)
    key_len = len(key)

    scores = [[0.0] * key_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(key_len):
            # Dot product
            score = sum(query[i][k] * key[j][k] for k in range(d_k))
            scores[i][j] = score / scale

    # Apply mask if provided
    if mask:
        for i in range(seq_len):
            for j in range(key_len):
                if mask[i][j] < 0:
                    scores[i][j] = float("-inf")

    # Softmax
    attention_weights = []
    for row in scores:
        max_val = max(row) if row else 0
        exp_row = [math.exp(x - max_val) if x != float("-inf") else 0 for x in row]
        sum_exp = sum(exp_row) or 1
        attention_weights.append([x / sum_exp for x in exp_row])

    # Compute output: attention_weights @ value
    value_dim = len(value[0]) if value and value[0] else 0
    output = [[0.0] * value_dim for _ in range(seq_len)]
    for i in range(seq_len):
        for k in range(value_dim):
            output[i][k] = sum(attention_weights[i][j] * value[j][k] for j in range(key_len))

    return output, attention_weights


class LinearLayer:
    """Simple linear transformation layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
    ):
        """
        Initialize linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            use_bias: Whether to use bias
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Initialize weights (Xavier/Glorot initialization)
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weights = [[self._random_normal() * scale for _ in range(in_features)] for _ in range(out_features)]

        if use_bias:
            self.bias = [0.0] * out_features
        else:
            self.bias = None

    def _random_normal(self) -> float:
        """Generate pseudo-random normal value using Box-Muller."""
        import random

        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        """
        Forward pass.

        Args:
            x: Input tensor (seq_len, in_features)

        Returns:
            Output tensor (seq_len, out_features)
        """
        seq_len = len(x)
        output = []

        for i in range(seq_len):
            row = []
            for j in range(self.out_features):
                val = sum(self.weights[j][k] * x[i][k] for k in range(self.in_features))
                if self.bias:
                    val += self.bias[j]
                row.append(val)
            output.append(row)

        return output


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_bias: Whether to use bias in projections
        """
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections
        self.q_proj = LinearLayer(embed_dim, embed_dim, use_bias)
        self.k_proj = LinearLayer(embed_dim, embed_dim, use_bias)
        self.v_proj = LinearLayer(embed_dim, embed_dim, use_bias)
        self.out_proj = LinearLayer(embed_dim, embed_dim, use_bias)

    def _split_heads(
        self,
        x: list[list[float]],
    ) -> list[list[list[float]]]:
        """
        Split tensor into multiple heads.

        Args:
            x: Input (seq_len, embed_dim)

        Returns:
            Output (num_heads, seq_len, head_dim)
        """
        seq_len = len(x)
        result = []

        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.head_dim
                end = start + self.head_dim
                head.append(x[i][start:end])
            result.append(head)

        return result

    def _combine_heads(
        self,
        heads: list[list[list[float]]],
    ) -> list[list[float]]:
        """
        Combine heads back into single tensor.

        Args:
            heads: Input (num_heads, seq_len, head_dim)

        Returns:
            Output (seq_len, embed_dim)
        """
        if not heads or not heads[0]:
            return []

        seq_len = len(heads[0])
        result = []

        for i in range(seq_len):
            row = []
            for h in range(self.num_heads):
                row.extend(heads[h][i])
            result.append(row)

        return result

    def forward(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
        mask: list[list[float]] | None = None,
        return_attention: bool = True,
    ) -> AttentionOutput:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (seq_len, embed_dim)
            key: Key tensor (seq_len, embed_dim)
            value: Value tensor (seq_len, embed_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            AttentionOutput
        """
        # Project queries, keys, values
        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        # Split into heads
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        # Compute attention for each head
        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            output, weights = scaled_dot_product_attention(
                q_heads[h],
                k_heads[h],
                v_heads[h],
                mask,
                self.dropout,
            )
            head_outputs.append(output)
            head_weights.append(weights)

        # Combine heads
        combined = self._combine_heads(head_outputs)

        # Final projection
        output = self.out_proj.forward(combined)

        # Average attention weights across heads
        if head_weights and head_weights[0]:
            seq_len = len(head_weights[0])
            key_len = len(head_weights[0][0]) if head_weights[0] else 0

            avg_weights = [[0.0] * key_len for _ in range(seq_len)]
            for h in range(self.num_heads):
                for i in range(seq_len):
                    for j in range(key_len):
                        avg_weights[i][j] += head_weights[h][i][j] / self.num_heads
        else:
            avg_weights = []

        attention_weights = AttentionWeights(
            weights=[avg_weights],  # Batch dimension
            head_weights=[head_weights] if return_attention else None,
        )

        return AttentionOutput(
            output=[output],  # Batch dimension
            attention_weights=attention_weights,
        )

    def get_correlation_matrix(
        self,
        attention_weights: AttentionWeights,
    ) -> list[list[float]]:
        """
        Extract correlation matrix from attention weights.

        The attention weights can be interpreted as a soft correlation
        matrix between assets.

        Args:
            attention_weights: Attention weights from forward pass

        Returns:
            Correlation matrix (num_assets, num_assets)
        """
        if not attention_weights.weights or not attention_weights.weights[0]:
            return []

        weights = attention_weights.weights[0]  # First batch
        num_assets = len(weights)

        # Symmetrize the attention weights
        correlation = [[0.0] * num_assets for _ in range(num_assets)]
        for i in range(num_assets):
            for j in range(num_assets):
                # Average bidirectional attention
                correlation[i][j] = (weights[i][j] + weights[j][i]) / 2

        return correlation


class PositionalEncoding:
    """
    Positional encoding for sequence data.

    Uses sinusoidal positional encoding as in "Attention Is All You Need".
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 1000,
    ):
        """
        Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Precompute positional encodings
        self.encodings = self._compute_encodings()

    def _compute_encodings(self) -> list[list[float]]:
        """Compute sinusoidal positional encodings."""
        encodings = []

        for pos in range(self.max_len):
            encoding = []
            for i in range(self.embed_dim):
                if i % 2 == 0:
                    # sin(pos / 10000^(2i/d))
                    val = math.sin(pos / math.pow(10000, i / self.embed_dim))
                else:
                    # cos(pos / 10000^(2i/d))
                    val = math.cos(pos / math.pow(10000, (i - 1) / self.embed_dim))
                encoding.append(val)
            encodings.append(encoding)

        return encodings

    def forward(
        self,
        x: list[list[float]],
    ) -> list[list[float]]:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (seq_len, embed_dim)

        Returns:
            Output with positional encoding added
        """
        seq_len = len(x)
        output = []

        for i in range(min(seq_len, self.max_len)):
            row = [x[i][j] + self.encodings[i][j] for j in range(self.embed_dim)]
            output.append(row)

        # If seq_len > max_len, just copy remaining without encoding
        for i in range(self.max_len, seq_len):
            output.append(x[i].copy())

        return output


class AssetEncoder:
    """
    Encodes asset features for attention-based processing.

    Combines:
    - Feature embedding
    - Positional encoding
    - Layer normalization
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        max_assets: int = 100,
    ):
        """
        Initialize asset encoder.

        Args:
            input_dim: Input feature dimension
            embed_dim: Output embedding dimension
            max_assets: Maximum number of assets
        """
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Feature embedding
        self.feature_proj = LinearLayer(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_assets)

    def forward(
        self,
        features: list[list[float]],
    ) -> list[list[float]]:
        """
        Encode asset features.

        Args:
            features: Asset features (num_assets, input_dim)

        Returns:
            Encoded features (num_assets, embed_dim)
        """
        # Project features
        embedded = self.feature_proj.forward(features)

        # Add positional encoding
        encoded = self.pos_encoding.forward(embedded)

        return encoded


@dataclass
class AttentionPPOConfig:
    """Configuration for attention-enhanced PPO."""

    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_asset_encoder: bool = True
    use_positional_encoding: bool = True


class AttentionBlock:
    """
    Single attention block with residual connection and normalization.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize attention block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.embed_dim = embed_dim

    def _layer_norm(
        self,
        x: list[list[float]],
        eps: float = 1e-6,
    ) -> list[list[float]]:
        """Apply layer normalization."""
        output = []
        for row in x:
            mean = sum(row) / len(row) if row else 0
            variance = sum((v - mean) ** 2 for v in row) / len(row) if row else 1
            std = math.sqrt(variance + eps)
            normalized = [(v - mean) / std for v in row]
            output.append(normalized)
        return output

    def forward(
        self,
        x: list[list[float]],
        mask: list[list[float]] | None = None,
    ) -> tuple[list[list[float]], AttentionWeights]:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention
        attn_output = self.attention.forward(x, x, x, mask)

        # Residual connection
        output = attn_output.output[0]  # Remove batch dim
        residual = [[output[i][j] + x[i][j] for j in range(self.embed_dim)] for i in range(len(x))]

        # Layer normalization
        normalized = self._layer_norm(residual)

        return normalized, attn_output.attention_weights


def create_attention_layer(
    embed_dim: int = 64,
    num_heads: int = 4,
    dropout: float = 0.1,
    use_bias: bool = True,
) -> MultiHeadAttention:
    """
    Factory function to create a MultiHeadAttention layer.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_bias: Whether to use bias

    Returns:
        Configured MultiHeadAttention
    """
    return MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_bias=use_bias,
    )


def create_asset_encoder(
    input_dim: int,
    embed_dim: int = 64,
    max_assets: int = 100,
) -> AssetEncoder:
    """
    Factory function to create an AssetEncoder.

    Args:
        input_dim: Input feature dimension
        embed_dim: Output embedding dimension
        max_assets: Maximum number of assets

    Returns:
        Configured AssetEncoder
    """
    return AssetEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        max_assets=max_assets,
    )


def get_attention_weights(
    attention_output: AttentionOutput,
    head_index: int | None = None,
) -> list[list[float]]:
    """
    Extract attention weights from an AttentionOutput.

    Helper function for visualization - extracts attention weights
    in a format suitable for heatmap display.

    Args:
        attention_output: Output from MultiHeadAttention.forward()
        head_index: Optional head index to get specific head weights.
                   If None, returns averaged weights across all heads.

    Returns:
        2D attention weights matrix (query_len, key_len)

    Example:
        >>> attention = create_attention_layer(embed_dim=64, num_heads=4)
        >>> output = attention.forward(query, key, value)
        >>> weights = get_attention_weights(output)  # Averaged weights
        >>> head_0_weights = get_attention_weights(output, head_index=0)
    """
    attn_weights = attention_output.attention_weights

    if head_index is not None:
        # Return specific head weights
        if attn_weights.head_weights and len(attn_weights.head_weights) > 0:
            batch_head_weights = attn_weights.head_weights[0]  # First batch
            if 0 <= head_index < len(batch_head_weights):
                return batch_head_weights[head_index]
        return []

    # Return averaged weights
    if attn_weights.weights and len(attn_weights.weights) > 0:
        return attn_weights.weights[0]  # First batch

    return []


def get_all_head_weights(
    attention_output: AttentionOutput,
) -> list[list[list[float]]]:
    """
    Extract all per-head attention weights from an AttentionOutput.

    Args:
        attention_output: Output from MultiHeadAttention.forward()

    Returns:
        Per-head attention weights (num_heads, query_len, key_len)

    Example:
        >>> attention = create_attention_layer(embed_dim=64, num_heads=4)
        >>> output = attention.forward(query, key, value)
        >>> all_heads = get_all_head_weights(output)  # 4 heads
        >>> print(f"Number of heads: {len(all_heads)}")
    """
    attn_weights = attention_output.attention_weights

    if attn_weights.head_weights and len(attn_weights.head_weights) > 0:
        return attn_weights.head_weights[0]  # First batch

    return []
