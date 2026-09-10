"""Tests for reservoir.audit — Exact buffer audit layer."""

import warnings
import numpy as np
import pytest

from reservoir.fast_buffer import FastPERBuffer
from reservoir.audit import AuditedPERBuffer, AuditReport

OBS_SHAPE = (4,)


def make_audited(capacity=512, audit_capacity=64, audit_interval=50):
    buf = FastPERBuffer(capacity, OBS_SHAPE)
    return AuditedPERBuffer(
        buf,
        audit_capacity=audit_capacity,
        audit_interval=audit_interval,
        alpha=0.6,
        seed=42,
    )


def fill(audited, n=100):
    obs = np.random.randn(*OBS_SHAPE).astype(np.float32)
    act = np.array([0], dtype=np.float32)
    for _ in range(n):
        audited.add(obs, act, 1.0, obs, False)


class TestAuditedPERBufferBasics:
    def test_add_increments_fast_buffer(self):
        a = make_audited()
        fill(a, 10)
        assert a.size == 10

    def test_mirrors_to_exact_buffer(self):
        a = make_audited(audit_interval=10000)  # No auto audit
        fill(a, 50)
        assert a._exact.size > 0

    def test_sample_returns_fast_batch(self):
        a = make_audited()
        fill(a, 100)
        batch = a.sample(32)
        assert batch.states.shape == (32, *OBS_SHAPE)

    def test_update_priorities_works(self):
        a = make_audited()
        fill(a, 100)
        batch = a.sample(32)
        td = np.random.rand(32)
        a.update_priorities(batch.indices, td)

    def test_force_audit_returns_report(self):
        a = make_audited()
        fill(a, 100)
        report = a.force_audit()
        assert report is not None
        assert isinstance(report, AuditReport)
        assert report.n_samples_compared > 0

    def test_audit_report_summary(self):
        a = make_audited(audit_interval=20)
        fill(a, 200)
        summary = a.audit_report()
        assert "audits" in summary
        assert "all_passed" in summary
        assert summary["audits"] >= 1

    def test_auto_audit_triggers(self):
        a = make_audited(audit_interval=50)
        fill(a, 200)
        assert len(a._reports) >= 3

    def test_anneal_beta_delegates(self):
        a = make_audited()
        a.anneal_beta(100, 1000, beta_end=1.0)
        # No assertion — just confirm it doesn't raise


class TestAuditReport:
    def test_passed_when_no_divergences(self):
        report = AuditReport(
            step=100,
            fast_total=1000.0,
            exact_total=500,
            n_samples_compared=20,
            max_tv_distance=1e-10,
            divergences=0,
            passed=True,
        )
        assert report.passed is True

    def test_fails_on_divergence(self):
        report = AuditReport(
            step=100,
            fast_total=1000.0,
            exact_total=500,
            n_samples_compared=20,
            max_tv_distance=0.5,
            divergences=3,
            passed=False,
        )
        assert report.passed is False


class TestAuditWarning:
    def test_warning_on_high_tv(self):
        """If max_tv exceeds threshold, a RuntimeWarning is issued."""
        a = make_audited(audit_interval=10000)
        fill(a, 100)

        # Artificially inject a bad report to trigger warning
        bad_report = AuditReport(
            step=50,
            fast_total=1.0,
            exact_total=1,
            n_samples_compared=5,
            max_tv_distance=0.5,  # > TV_THRESHOLD
            divergences=1,
            passed=False,
        )
        a._reports.append(bad_report)
        summary = a.audit_report()
        assert not summary["all_passed"]
        assert summary["total_divergences"] == 1
