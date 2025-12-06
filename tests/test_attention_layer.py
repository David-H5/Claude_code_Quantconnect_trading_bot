"""
Tests for Multi-Head Attention Layer (UPGRADE-010 Sprint 2)

Tests attention mechanisms for PPO portfolio optimization.
"""

import pytest

from models.attention_layer import (
    AssetEncoder,
    AttentionBlock,
    AttentionConfig,
    AttentionOutput,
    AttentionPPOConfig,
    AttentionType,
    AttentionWeights,
    LinearLayer,
    MultiHeadAttention,
    PositionalEncoding,
    create_asset_encoder,
    create_attention_layer,
    scaled_dot_product_attention,
)


class TestAttentionType:
    """Tests for AttentionType enum."""

    @pytest.mark.unit
    def test_attention_types_exist(self):
        """Test all attention types exist."""
        assert AttentionType.SELF is not None
        assert AttentionType.CROSS is not None
        assert AttentionType.MASKED is not None


class TestAttentionConfig:
    """Tests for AttentionConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration."""
        config = AttentionConfig()

        assert config.embed_dim == 64
        assert config.num_heads == 4
        assert config.dropout == 0.1

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = AttentionConfig(
            embed_dim=128,
            num_heads=8,
            dropout=0.2,
        )

        assert config.embed_dim == 128
        assert config.num_heads == 8


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    @pytest.mark.unit
    def test_basic_attention(self):
        """Test basic attention computation."""
        # Simple 2x2 case
        query = [[1.0, 0.0], [0.0, 1.0]]
        key = [[1.0, 0.0], [0.0, 1.0]]
        value = [[1.0, 2.0], [3.0, 4.0]]

        output, weights = scaled_dot_product_attention(query, key, value)

        assert len(output) == 2
        assert len(output[0]) == 2
        assert len(weights) == 2

    @pytest.mark.unit
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        query = [[1.0, 0.5, 0.2], [0.3, 0.8, 0.1]]
        key = [[0.5, 0.5, 0.5], [0.2, 0.3, 0.4], [0.1, 0.1, 0.1]]
        value = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

        _, weights = scaled_dot_product_attention(query, key, value)

        for row in weights:
            assert sum(row) == pytest.approx(1.0, rel=0.01)

    @pytest.mark.unit
    def test_attention_with_mask(self):
        """Test attention with masking."""
        query = [[1.0, 0.0], [0.0, 1.0]]
        key = [[1.0, 0.0], [0.0, 1.0]]
        value = [[1.0, 2.0], [3.0, 4.0]]
        mask = [[0.0, -float("inf")], [0.0, 0.0]]  # Mask second key for first query

        output, weights = scaled_dot_product_attention(query, key, value, mask)

        # First query should only attend to first key
        assert weights[0][1] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.unit
    def test_attention_empty_input(self):
        """Test attention with empty input."""
        output, weights = scaled_dot_product_attention([], [], [])

        assert output == []
        assert weights == []


class TestLinearLayer:
    """Tests for LinearLayer."""

    @pytest.mark.unit
    def test_linear_creation(self):
        """Test linear layer creation."""
        layer = LinearLayer(10, 5)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert len(layer.weights) == 5
        assert len(layer.weights[0]) == 10

    @pytest.mark.unit
    def test_linear_forward(self):
        """Test linear forward pass."""
        layer = LinearLayer(3, 2)

        x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        output = layer.forward(x)

        assert len(output) == 2
        assert len(output[0]) == 2

    @pytest.mark.unit
    def test_linear_no_bias(self):
        """Test linear without bias."""
        layer = LinearLayer(3, 2, use_bias=False)

        assert layer.bias is None


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    @pytest.fixture
    def attention(self) -> MultiHeadAttention:
        """Create attention layer for testing."""
        return MultiHeadAttention(embed_dim=8, num_heads=2)

    @pytest.mark.unit
    def test_attention_creation(self, attention):
        """Test attention creation."""
        assert attention.embed_dim == 8
        assert attention.num_heads == 2
        assert attention.head_dim == 4

    @pytest.mark.unit
    def test_attention_invalid_dims(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            MultiHeadAttention(embed_dim=10, num_heads=3)  # 10 not divisible by 3

    @pytest.mark.unit
    def test_forward_self_attention(self, attention):
        """Test self-attention forward pass."""
        # (seq_len=3, embed_dim=8)
        x = [[float(i + j) / 10 for j in range(8)] for i in range(3)]

        output = attention.forward(x, x, x)

        assert isinstance(output, AttentionOutput)
        assert len(output.output) == 1  # Batch dim
        assert len(output.output[0]) == 3  # Seq len
        assert len(output.output[0][0]) == 8  # Embed dim

    @pytest.mark.unit
    def test_forward_returns_attention_weights(self, attention):
        """Test that forward returns attention weights."""
        x = [[float(i + j) / 10 for j in range(8)] for i in range(3)]

        output = attention.forward(x, x, x, return_attention=True)

        assert output.attention_weights is not None
        assert len(output.attention_weights.weights) == 1

    @pytest.mark.unit
    def test_get_correlation_matrix(self, attention):
        """Test correlation matrix extraction."""
        x = [[float(i + j) / 10 for j in range(8)] for i in range(3)]

        output = attention.forward(x, x, x)
        correlation = attention.get_correlation_matrix(output.attention_weights)

        assert len(correlation) == 3
        assert len(correlation[0]) == 3

        # Check symmetry
        for i in range(3):
            for j in range(3):
                assert correlation[i][j] == pytest.approx(correlation[j][i], rel=0.01)

    @pytest.mark.unit
    def test_split_and_combine_heads(self, attention):
        """Test head splitting and combining."""
        x = [[float(i) for i in range(8)] for _ in range(2)]

        heads = attention._split_heads(x)
        assert len(heads) == 2  # num_heads
        assert len(heads[0]) == 2  # seq_len
        assert len(heads[0][0]) == 4  # head_dim

        combined = attention._combine_heads(heads)
        assert len(combined) == 2  # seq_len
        assert len(combined[0]) == 8  # embed_dim


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    @pytest.fixture
    def pos_encoding(self) -> PositionalEncoding:
        """Create positional encoding for testing."""
        return PositionalEncoding(embed_dim=8, max_len=100)

    @pytest.mark.unit
    def test_encoding_creation(self, pos_encoding):
        """Test encoding creation."""
        assert pos_encoding.embed_dim == 8
        assert pos_encoding.max_len == 100
        assert len(pos_encoding.encodings) == 100

    @pytest.mark.unit
    def test_encoding_forward(self, pos_encoding):
        """Test adding positional encoding."""
        x = [[0.0] * 8 for _ in range(10)]

        output = pos_encoding.forward(x)

        assert len(output) == 10
        # Output should be different from input (encoding added)
        assert output[0] != x[0]

    @pytest.mark.unit
    def test_different_positions_have_different_encodings(self, pos_encoding):
        """Test that different positions have different encodings."""
        assert pos_encoding.encodings[0] != pos_encoding.encodings[1]

    @pytest.mark.unit
    def test_encoding_values_bounded(self, pos_encoding):
        """Test encoding values are bounded."""
        for encoding in pos_encoding.encodings:
            for val in encoding:
                assert -1.0 <= val <= 1.0


class TestAssetEncoder:
    """Tests for AssetEncoder."""

    @pytest.fixture
    def encoder(self) -> AssetEncoder:
        """Create encoder for testing."""
        return AssetEncoder(input_dim=5, embed_dim=8)

    @pytest.mark.unit
    def test_encoder_creation(self, encoder):
        """Test encoder creation."""
        assert encoder.input_dim == 5
        assert encoder.embed_dim == 8

    @pytest.mark.unit
    def test_encoder_forward(self, encoder):
        """Test encoder forward pass."""
        # 3 assets, 5 features each
        features = [[float(i + j) / 10 for j in range(5)] for i in range(3)]

        output = encoder.forward(features)

        assert len(output) == 3  # num_assets
        assert len(output[0]) == 8  # embed_dim


class TestAttentionBlock:
    """Tests for AttentionBlock."""

    @pytest.fixture
    def block(self) -> AttentionBlock:
        """Create attention block for testing."""
        return AttentionBlock(embed_dim=8, num_heads=2)

    @pytest.mark.unit
    def test_block_creation(self, block):
        """Test block creation."""
        assert block.embed_dim == 8

    @pytest.mark.unit
    def test_block_forward(self, block):
        """Test block forward pass with residual."""
        x = [[float(i + j) / 10 for j in range(8)] for i in range(3)]

        output, weights = block.forward(x)

        assert len(output) == 3
        assert len(output[0]) == 8

    @pytest.mark.unit
    def test_layer_norm(self, block):
        """Test layer normalization."""
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]

        normalized = block._layer_norm(x)

        # Mean should be ~0, std should be ~1
        mean = sum(normalized[0]) / len(normalized[0])
        assert mean == pytest.approx(0.0, abs=0.01)


class TestAttentionWeights:
    """Tests for AttentionWeights dataclass."""

    @pytest.mark.unit
    def test_weights_creation(self):
        """Test creating attention weights."""
        weights = AttentionWeights(
            weights=[[[0.5, 0.5], [0.3, 0.7]]],
        )

        assert len(weights.weights) == 1
        assert weights.head_weights is None


class TestAttentionOutput:
    """Tests for AttentionOutput dataclass."""

    @pytest.mark.unit
    def test_output_creation(self):
        """Test creating attention output."""
        output = AttentionOutput(
            output=[[[1.0, 2.0], [3.0, 4.0]]],
            attention_weights=AttentionWeights(weights=[[[0.5, 0.5]]]),
        )

        assert len(output.output) == 1
        assert output.correlation_matrix is None


class TestAttentionPPOConfig:
    """Tests for AttentionPPOConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default PPO config."""
        config = AttentionPPOConfig()

        assert config.embed_dim == 64
        assert config.num_heads == 4
        assert config.num_layers == 2

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom PPO config."""
        config = AttentionPPOConfig(
            embed_dim=128,
            num_layers=4,
        )

        assert config.embed_dim == 128
        assert config.num_layers == 4


class TestCreateAttentionLayer:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test factory with defaults."""
        attention = create_attention_layer()

        assert attention.embed_dim == 64
        assert attention.num_heads == 4

    @pytest.mark.unit
    def test_create_with_custom(self):
        """Test factory with custom values."""
        attention = create_attention_layer(
            embed_dim=128,
            num_heads=8,
        )

        assert attention.embed_dim == 128
        assert attention.num_heads == 8


class TestCreateAssetEncoder:
    """Tests for asset encoder factory."""

    @pytest.mark.unit
    def test_create_encoder(self):
        """Test creating encoder."""
        encoder = create_asset_encoder(
            input_dim=10,
            embed_dim=32,
        )

        assert encoder.input_dim == 10
        assert encoder.embed_dim == 32


class TestIntegration:
    """Integration tests for attention components."""

    @pytest.mark.unit
    def test_full_attention_pipeline(self):
        """Test complete attention pipeline."""
        # Create components
        encoder = create_asset_encoder(input_dim=5, embed_dim=8)
        attention = create_attention_layer(embed_dim=8, num_heads=2)

        # Input: 4 assets with 5 features each
        asset_features = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8],
        ]

        # Encode
        encoded = encoder.forward(asset_features)
        assert len(encoded) == 4
        assert len(encoded[0]) == 8

        # Self-attention
        output = attention.forward(encoded, encoded, encoded)

        assert len(output.output[0]) == 4  # Same number of assets
        assert len(output.output[0][0]) == 8  # Same embed dim

        # Get correlations
        correlations = attention.get_correlation_matrix(output.attention_weights)
        assert len(correlations) == 4

    @pytest.mark.unit
    def test_multi_layer_attention(self):
        """Test stacking attention blocks."""
        blocks = [AttentionBlock(embed_dim=8, num_heads=2) for _ in range(3)]

        x = [[float(i + j) / 10 for j in range(8)] for i in range(4)]

        # Pass through multiple blocks
        for block in blocks:
            x, _ = block.forward(x)

        assert len(x) == 4
        assert len(x[0]) == 8

    @pytest.mark.unit
    def test_attention_preserves_sequence_length(self):
        """Test that attention preserves sequence length."""
        attention = create_attention_layer(embed_dim=16, num_heads=4)

        for seq_len in [1, 5, 10, 20]:
            x = [[float(i) / 10 for i in range(16)] for _ in range(seq_len)]
            output = attention.forward(x, x, x)

            assert len(output.output[0]) == seq_len
