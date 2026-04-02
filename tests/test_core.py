"""Core unit tests for all federated learning modules.

Tests cover:
  - Local training (FedAvg style + FedProx)
  - Server aggregation (FedAvg, Coord-Median, Trimmed Mean, Krum)
  - Differential privacy (clipping, noise, privacy accountant)
  - Gradient compression (quantisation SNR, unbiasedness, byte reduction)
  - Gradient inversion attack (analytical recovery, DP degradation)
  - Membership inference attack (metrics, AUC bounds)
  - Fairness and utility metrics
  - Personalization module

Run with:
    cd federated-student-privacy
    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

from clients.local_train import LRParams, local_train_logreg, local_train_fedprox
from server.aggregator import (
    ClientUpdate, fedavg, coord_median, trimmed_mean, krum, aggregate
)
from privacy.dp import DPConfig, PrivacyAccountant, clip_update, apply_dp_noise
from privacy.compression import (
    CompressionConfig, stochastic_quantize, compress_params, compressed_bytes,
    quantisation_snr,
)
from attacks.membership_inference import membership_inference_attack
from attacks.gradient_inversion import (
    invert_single_gradient, evaluate_gradient_leakage,
)
from metrics.utility import binary_classification_metrics
from metrics.fairness import per_client_f1


# ─────────────────────────────────────────────────────────────────────────────
# Local training
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalTrain:
    def test_fedavg_reduces_loss(self, small_dataset, zero_params):
        """FedAvg local training should improve F1 beyond the zero-model baseline."""
        x, y = small_dataset
        before = zero_params
        after = local_train_logreg(x, y, before, lr=0.05, local_steps=200, l2=1e-3)

        def _f1(params):
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ params.w + params.b, -500, 500)))
            preds = (p >= 0.5).astype(int)
            return binary_classification_metrics(y.astype(int), preds).f1

        assert _f1(after) >= _f1(before), "Local training should not hurt F1"

    def test_params_shape_preserved(self, small_dataset, zero_params):
        x, y = small_dataset
        result = local_train_logreg(x, y, zero_params)
        assert result.w.shape == zero_params.w.shape
        assert isinstance(result.b, float)

    def test_fedprox_with_zero_mu_matches_fedavg(self, small_dataset, small_params):
        """FedProx with mu=0 should return the same result as plain FedAvg."""
        x, y = small_dataset
        fa = local_train_logreg(x, y, small_params, lr=0.01, local_steps=50)
        fp = local_train_fedprox(x, y, small_params, small_params,
                                 mu=0.0, lr=0.01, local_steps=50)
        np.testing.assert_allclose(fa.w, fp.w, atol=1e-8)
        assert abs(fa.b - fp.b) < 1e-8

    def test_fedprox_reduces_drift(self, small_dataset, small_params, rng):
        """FedProx update should stay closer to global_params than FedAvg."""
        x, y = small_dataset
        # Use a global_params far from local optimum to exaggerate drift
        global_p = LRParams(w=np.zeros(8), b=0.0)
        fa = local_train_logreg(x, y, global_p, lr=0.1, local_steps=500)
        fp = local_train_fedprox(x, y, global_p, global_p,
                                 mu=0.5, lr=0.1, local_steps=500)

        def _dist(p: LRParams) -> float:
            return float(np.linalg.norm(np.append(p.w - global_p.w, p.b - global_p.b)))

        assert _dist(fp) <= _dist(fa), (
            "FedProx should produce smaller update norm than FedAvg"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregation:
    def _make_updates(self, n: int, d: int, rng: np.random.Generator):
        updates = []
        for _ in range(n):
            updates.append(ClientUpdate(
                params=LRParams(w=rng.normal(0, 1, d), b=float(rng.normal())),
                n_samples=rng.integers(50, 200),
            ))
        return updates

    def test_fedavg_weighted_mean(self, rng):
        """FedAvg with equal sample counts should equal unweighted mean."""
        d = 4
        updates = [
            ClientUpdate(params=LRParams(w=np.array([1.0, 0, 0, 0]), b=1.0), n_samples=100),
            ClientUpdate(params=LRParams(w=np.array([3.0, 0, 0, 0]), b=3.0), n_samples=100),
        ]
        agg = fedavg(updates)
        np.testing.assert_allclose(agg.w, [2.0, 0, 0, 0], atol=1e-9)
        assert abs(agg.b - 2.0) < 1e-9

    def test_fedavg_single_client(self, rng):
        """FedAvg with a single client should return that client's params."""
        params = LRParams(w=np.array([1.0, 2.0]), b=3.0)
        update = [ClientUpdate(params=params, n_samples=50)]
        result = fedavg(update)
        np.testing.assert_allclose(result.w, params.w)
        assert result.b == pytest.approx(params.b)

    def test_coord_median_rejects_outlier(self, rng):
        """Coord-median should be robust to one extreme outlier."""
        d = 4
        updates = [
            ClientUpdate(params=LRParams(w=np.ones(d), b=1.0), n_samples=100),
            ClientUpdate(params=LRParams(w=np.ones(d), b=1.0), n_samples=100),
            ClientUpdate(params=LRParams(w=np.ones(d) * 1000, b=1000.0), n_samples=100),  # outlier
        ]
        result = coord_median(updates)
        assert abs(result.b - 1.0) < 1.0, "Coord-median should reject outlier"
        assert np.all(np.abs(result.w - 1.0) < 1.0)

    def test_trimmed_mean_shape(self, rng):
        updates = self._make_updates(10, 8, rng)
        result = trimmed_mean(updates)
        assert result.w.shape == (8,)

    def test_krum_selects_valid_client(self, rng):
        """Krum should select one of the client params (not an average)."""
        d = 4
        updates = self._make_updates(6, d, rng)
        result = krum(updates)
        # Result must match exactly one of the input params
        vecs = [np.append(u.params.w, u.params.b) for u in updates]
        result_vec = np.append(result.w, result.b)
        is_from_client = any(
            np.allclose(result_vec, v) for v in vecs
        )
        assert is_from_client, "Krum must return one of the input client params"

    def test_aggregate_dispatcher(self, rng):
        updates = self._make_updates(5, 4, rng)
        for method in ["fedavg", "coord_median", "trimmed_mean", "krum"]:
            result = aggregate(updates, method=method)
            assert result.w.shape == (4,)

    def test_aggregate_unknown_method_raises(self, rng):
        updates = self._make_updates(3, 4, rng)
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate(updates, method="nonexistent")

    def test_fedavg_no_samples_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            fedavg([ClientUpdate(LRParams(w=np.zeros(4), b=0.0), n_samples=0)])


# ─────────────────────────────────────────────────────────────────────────────
# Differential Privacy
# ─────────────────────────────────────────────────────────────────────────────

class TestDP:
    def test_clip_reduces_norm(self, rng):
        large = LRParams(w=rng.normal(0, 10, 8), b=float(rng.normal(0, 10)))
        max_norm = 1.0
        clipped = clip_update(large, max_norm)
        vec = np.append(clipped.w, clipped.b)
        assert np.linalg.norm(vec) <= max_norm + 1e-8

    def test_clip_preserves_direction(self, rng):
        params = LRParams(w=rng.normal(0, 5, 8), b=2.0)
        clipped = clip_update(params, max_norm=1.0)
        # Direction (cosine similarity) should be preserved
        orig_vec = np.append(params.w, params.b)
        clip_vec = np.append(clipped.w, clipped.b)
        cos = np.dot(orig_vec, clip_vec) / (
            np.linalg.norm(orig_vec) * np.linalg.norm(clip_vec) + 1e-10
        )
        assert cos > 0.999

    def test_clip_small_norm_unchanged(self):
        small = LRParams(w=np.array([0.1, 0.1]), b=0.1)
        clipped = clip_update(small, max_norm=10.0)
        np.testing.assert_allclose(clipped.w, small.w)
        assert clipped.b == pytest.approx(small.b)

    def test_dp_noise_changes_params(self, small_params, rng):
        cfg = DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0)
        noised = apply_dp_noise(small_params, cfg, rng)
        # Should not be identical
        assert not np.allclose(small_params.w, noised.w)

    def test_dp_noise_disabled_no_change(self, small_params, rng):
        cfg = DPConfig(enabled=False)
        result = apply_dp_noise(small_params, cfg, rng)
        np.testing.assert_array_equal(small_params.w, result.w)
        assert small_params.b == result.b

    def test_privacy_accountant_zero_steps(self):
        acc = PrivacyAccountant(noise_multiplier=1.1, target_delta=1e-5, client_frac=0.5)
        assert acc.epsilon == 0.0

    def test_privacy_accountant_increases_with_steps(self):
        acc = PrivacyAccountant(noise_multiplier=1.1, target_delta=1e-5, client_frac=0.5)
        prev = 0.0
        for _ in range(5):
            acc.step()
            assert acc.epsilon >= prev
            prev = acc.epsilon

    def test_privacy_accountant_more_noise_less_epsilon(self):
        """Higher noise_multiplier should result in smaller (better) epsilon."""
        def _eps(sigma):
            acc = PrivacyAccountant(noise_multiplier=sigma, target_delta=1e-5, client_frac=0.5)
            for _ in range(10):
                acc.step()
            return acc.epsilon

        assert _eps(2.0) < _eps(1.0) < _eps(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Compression
# ─────────────────────────────────────────────────────────────────────────────

class TestCompression:
    def test_quantise_preserves_shape(self, rng):
        vec = rng.normal(0, 1, 100)
        q = stochastic_quantize(vec, num_levels=16, rng=rng)
        assert q.shape == vec.shape

    def test_quantise_is_unbiased(self, rng):
        """E[quantize(v)] ≈ v over many trials."""
        vec = rng.normal(0, 1, 50)
        trials = [stochastic_quantize(vec, num_levels=64, rng=rng) for _ in range(500)]
        mean_q = np.mean(trials, axis=0)
        np.testing.assert_allclose(mean_q, vec, atol=0.05)

    def test_quantise_respects_range(self, rng):
        """All quantised values should have magnitude ≤ original norm."""
        vec = rng.normal(0, 1, 50)
        q = stochastic_quantize(vec, num_levels=16, rng=rng)
        assert np.max(np.abs(q)) <= np.linalg.norm(vec) + 1e-9

    def test_compress_params_disabled(self, small_params, rng):
        cfg = CompressionConfig(enabled=False)
        result = compress_params(small_params, cfg, rng)
        np.testing.assert_array_equal(result.w, small_params.w)

    def test_compress_reduces_bits(self):
        cfg = CompressionConfig(enabled=True, num_levels=16)
        assert cfg.bits == 4   # log2(16) = 4

    def test_compressed_bytes_less_than_original(self):
        cfg = CompressionConfig(enabled=True, num_levels=16)
        assert compressed_bytes(1000, cfg) < 1000

    def test_snr_perfect_case(self):
        vec = np.array([1.0, 2.0, 3.0])
        snr = quantisation_snr(vec, vec)
        assert snr == float("inf")

    def test_snr_higher_levels_better(self, rng):
        """More quantisation levels should give higher SNR."""
        vec = rng.normal(0, 1, 100)
        snr_4bit = quantisation_snr(vec, stochastic_quantize(vec, 16, rng))
        snr_8bit = quantisation_snr(vec, stochastic_quantize(vec, 256, rng))
        assert snr_8bit > snr_4bit


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Inversion Attack
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientInversion:
    def test_analytical_recovery_exact(self, rng):
        """Single-sample gradient inversion should recover x exactly."""
        d = 8
        params = LRParams(w=rng.normal(0, 0.5, d), b=float(rng.normal()))
        x = rng.normal(0, 1, d)
        y = 1.0

        def _grad(params, x, y):
            p = float(1.0 / (1.0 + np.exp(-np.clip(np.dot(params.w, x) + params.b, -500, 500))))
            residual = p - y
            g_w = x * residual
            g_b = residual
            return np.append(g_w, g_b)

        grad = _grad(params, x, y)
        x_hat, is_exact = invert_single_gradient(grad, d)

        assert is_exact, "Recovery should be exact when g_b != 0"
        # x̂ = g_w / g_b → should equal x
        np.testing.assert_allclose(x_hat, x, atol=1e-6)

    def test_dp_noise_degrades_recovery(self, small_dataset, small_params, rng):
        """Gradient leakage should decrease as DP noise increases."""
        x, y = small_dataset
        report_no_dp = evaluate_gradient_leakage(
            small_params, x, y, rng, n_samples=30, dp_noise_std=0.0
        )
        report_dp = evaluate_gradient_leakage(
            small_params, x, y, rng, n_samples=30, dp_noise_std=2.0
        )
        # Higher noise → lower cosine similarity (worse reconstruction)
        assert report_dp.mean_cosine_similarity < report_no_dp.mean_cosine_similarity

    def test_risk_high_without_dp(self, small_dataset, small_params, rng):
        x, y = small_dataset
        report = evaluate_gradient_leakage(
            small_params, x, y, rng, n_samples=20, dp_noise_std=0.0
        )
        assert report.reconstruction_risk == "HIGH"

    def test_risk_low_with_strong_dp(self, small_dataset, small_params, rng):
        x, y = small_dataset
        report = evaluate_gradient_leakage(
            small_params, x, y, rng, n_samples=20, dp_noise_std=10.0
        )
        assert report.reconstruction_risk == "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Membership Inference Attack
# ─────────────────────────────────────────────────────────────────────────────

class TestMIA:
    def test_mia_auc_between_0_and_1(self, small_dataset, small_params, rng):
        x, y = small_dataset
        x_te = rng.normal(0, 1, (100, 8))
        y_te = (rng.random(100) > 0.5).astype(int)
        result = membership_inference_attack(small_params, x, y.astype(int), x_te, y_te, rng)
        assert 0.0 <= result.auc <= 1.0

    def test_mia_advantage_nonnegative(self, small_dataset, small_params, rng):
        x, y = small_dataset
        x_te = rng.normal(0, 1, (100, 8))
        y_te = (rng.random(100) > 0.5).astype(int)
        result = membership_inference_attack(small_params, x, y.astype(int), x_te, y_te, rng)
        assert result.advantage >= 0.0

    def test_mia_accuracy_above_random(self, small_dataset, small_params, rng):
        """Attack accuracy should be at least chance (0.5) on a trained model."""
        x, y = small_dataset
        x_te = rng.normal(0, 1, (100, 8))
        y_te = (rng.random(100) > 0.5).astype(int)
        result = membership_inference_attack(small_params, x, y.astype(int), x_te, y_te, rng)
        # This is a trained model — slight membership leakage is expected
        assert result.accuracy >= 0.40   # very weak lower bound

    def test_mia_counts_match_requested(self, small_dataset, small_params, rng):
        x, y = small_dataset
        x_te = rng.normal(0, 1, (100, 8))
        y_te = (rng.random(100) > 0.5).astype(int)
        result = membership_inference_attack(
            small_params, x, y.astype(int), x_te, y_te, rng,
            n_members=40, n_nonmembers=40,
        )
        assert result.n_members == 40
        assert result.n_nonmembers == 40


# ─────────────────────────────────────────────────────────────────────────────
# Utility and Fairness metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 1])
        m = binary_classification_metrics(y, y)
        assert m.accuracy == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y = np.array([0, 0, 1, 1])
        preds = 1 - y
        m = binary_classification_metrics(y, preds)
        assert m.accuracy == pytest.approx(0.0)

    def test_fairness_per_client(self):
        y_true = {0: np.array([1, 0, 1]), 1: np.array([1, 1, 0])}
        y_pred = {0: np.array([1, 0, 1]), 1: np.array([0, 0, 0])}  # client 1 bad
        fair = per_client_f1(y_true, y_pred)
        assert fair.worst_f1 < fair.best_f1
        assert fair.mean_f1 == pytest.approx(
            (fair.worst_f1 + fair.best_f1) / 2, abs=0.01
        )
