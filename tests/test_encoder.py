"""Tests for encoder modules (CNNEncoder and DNNEncoder)."""

import os
import tempfile

import numpy as np
import pytest
import torch

from lsme import LSME, SignatureEncoder, CNNEncoder, DNNEncoder


class TestCNNEncoderBasic:
    """Test CNN encoder basic functionality."""

    def test_cnn_encoder_basic(self, lsme_result_karate):
        """Test basic CNN encoding workflow."""
        encoder = CNNEncoder(
            embedding_dim=32,
            max_matrix_size=32,
            num_epochs=20,
            verbose=False,
            random_state=42
        )

        embeddings = encoder.fit_transform(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        # Verify output structure
        assert isinstance(embeddings, dict), "Embeddings should be dict"
        assert len(embeddings) == 34, "Should have embedding per node (34 nodes)"

        for node_id, emb in embeddings.items():
            assert isinstance(emb, np.ndarray), "Embedding should be numpy array"
            assert emb.shape == (32,), f"Embedding shape should be (32,), got {emb.shape}"


class TestDNNEncoderBasic:
    """Test DNN encoder basic functionality."""

    def test_dnn_encoder_basic(self, lsme_result_karate):
        """Test basic DNN encoding workflow."""
        encoder = DNNEncoder(
            embedding_dim=32,
            max_matrix_size=32,
            hidden_dims=[256, 128, 64],
            num_epochs=20,
            verbose=False,
            random_state=42
        )

        embeddings = encoder.fit_transform(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        # Verify output structure
        assert isinstance(embeddings, dict), "Embeddings should be dict"
        assert len(embeddings) == 34, "Should have embedding per node (34 nodes)"

        for node_id, emb in embeddings.items():
            assert isinstance(emb, np.ndarray), "Embedding should be numpy array"
            assert emb.shape == (32,), f"Embedding shape should be (32,), got {emb.shape}"


class TestEncodeDecode:
    """Test encode-decode roundtrip for both encoders."""

    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_encode_decode(self, encoder_cls, lsme_result_karate):
        """Test encode-decode roundtrip."""
        encoder = encoder_cls(
            embedding_dim=64,
            num_epochs=30,
            verbose=False,
            random_state=42
        )
        encoder.fit(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        # Encode then decode
        embeddings = encoder.encode(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )
        reconstructed = encoder.decode(embeddings)

        # Verify reconstruction shapes
        for node_id in lsme_result_karate['signature_matrices']:
            recon = reconstructed[node_id]
            assert recon.shape[0] == recon.shape[1], "Reconstruction should be square"
            assert recon.min() >= 0.0, "Reconstruction values should be >= 0"
            assert recon.max() <= 1.0, "Reconstruction values should be <= 1"


class TestVariableSizes:
    """Test handling of variable matrix sizes."""

    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_variable_sizes(self, encoder_cls, lsme_result_star):
        """Test variable matrix sizes handling."""
        sizes = [info['total_nodes'] for info in lsme_result_star['layer_info'].values()]
        assert len(set(sizes)) > 1, "Test graph should have varying neighborhood sizes"

        encoder = encoder_cls(
            embedding_dim=16,
            max_matrix_size=32,
            num_epochs=20,
            verbose=False,
            random_state=42
        )
        embeddings = encoder.fit_transform(
            lsme_result_star['signature_matrices'],
            lsme_result_star['layer_info']
        )

        for emb in embeddings.values():
            assert emb.shape == (16,), "All embeddings should have dim 16"


class TestReproducibility:
    """Test encoder reproducibility."""

    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_reproducibility(self, encoder_cls, lsme_result_karate):
        """Test reproducibility with random_state."""
        encoder1 = encoder_cls(embedding_dim=32, num_epochs=20, verbose=False, random_state=123)
        emb1 = encoder1.fit_transform(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        encoder2 = encoder_cls(embedding_dim=32, num_epochs=20, verbose=False, random_state=123)
        emb2 = encoder2.fit_transform(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        for node in emb1:
            assert np.allclose(emb1[node], emb2[node], atol=1e-5), \
                f"Embeddings should match for node {node}"


class TestSaveLoad:
    """Test model save/load functionality."""

    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_save_load(self, encoder_cls, lsme_result_karate, tmp_path):
        """Test model save/load for both encoders."""
        encoder = encoder_cls(embedding_dim=32, num_epochs=20, verbose=False, random_state=42)
        encoder.fit(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )
        emb_before = encoder.encode(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        path = tmp_path / f"{encoder_cls.__name__.lower()}_encoder.pt"
        encoder.save(str(path))

        loaded_encoder = encoder_cls.load(str(path))
        emb_after = loaded_encoder.encode(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        for node in emb_before:
            assert np.allclose(emb_before[node], emb_after[node]), \
                "Embeddings should match after load"


class TestReconstructionError:
    """Test reconstruction error computation."""

    @pytest.mark.slow
    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_reconstruction_error(self, encoder_cls, lsme_result_karate):
        """Test reconstruction error computation."""
        encoder = encoder_cls(embedding_dim=64, num_epochs=50, verbose=False, random_state=42)
        encoder.fit(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        errors = encoder.reconstruction_error(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        assert len(errors) == 34, "Should have error per node"
        for node_id, error in errors.items():
            assert isinstance(error, float), "Error should be float"
            assert error >= 0, "MSE should be non-negative"


class TestGPUSupport:
    """Test GPU support."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_gpu_support(self, encoder_cls, lsme_result_karate):
        """Test GPU training support."""
        encoder = encoder_cls(embedding_dim=32, num_epochs=10, device='cuda', verbose=True)
        embeddings = encoder.fit_transform(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        assert len(embeddings) == 34


class TestSmallGraph:
    """Test small graph edge case."""

    @pytest.mark.parametrize("encoder_cls", [CNNEncoder, DNNEncoder])
    def test_small_graph(self, encoder_cls, lsme_result_path):
        """Test small graph edge case."""
        encoder = encoder_cls(
            embedding_dim=8,
            max_matrix_size=16,
            num_epochs=20,
            verbose=False,
            random_state=42
        )
        embeddings = encoder.fit_transform(
            lsme_result_path['signature_matrices'],
            lsme_result_path['layer_info']
        )

        assert len(embeddings) == 3, "Should have 3 embeddings"
        for emb in embeddings.values():
            assert emb.shape == (8,), "Embedding dimension should be 8"


class TestAliases:
    """Test encoder aliases."""

    def test_signature_encoder_alias(self, lsme_result_karate):
        """Test SignatureEncoder is alias for CNNEncoder."""
        assert SignatureEncoder is CNNEncoder, "SignatureEncoder should be CNNEncoder"

        encoder = SignatureEncoder(
            embedding_dim=16, num_epochs=10, verbose=False, random_state=42
        )
        embeddings = encoder.fit_transform(
            lsme_result_karate['signature_matrices'],
            lsme_result_karate['layer_info']
        )

        assert len(embeddings) == 34
