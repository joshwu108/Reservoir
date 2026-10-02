"""Tests for the TRL import guard in ``reservoir.integrations._trl_compat``.

The guard is the one place the integration touches TRL's private surface
by name, so these tests pin its behaviour: a missing TRL is an
``ImportError`` that says how to install the pinned version, a TRL whose
``GRPOTrainer`` has lost an attribute the adapter relies on is an
``ImportError`` naming that attribute, and an untested TRL version is a
warning rather than a failure. Fake ``trl`` and ``transformers`` modules
are installed in ``sys.modules`` so the tests run without TRL.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from reservoir.integrations import _trl_compat
from reservoir.integrations._trl_compat import (
    PINNED_TRL_VERSION,
    REQUIRED_TRAINER_ATTRIBUTES,
    TESTED_TRL_VERSIONS,
    require_trl,
)


def _fake_trainer_class(*attributes: str) -> type:
    """A stand-in ``GRPOTrainer`` exposing exactly ``attributes``."""
    namespace = {name: (lambda self, *a, **k: None) for name in attributes}
    return type("GRPOTrainer", (), namespace)


def _install_fake_trl(monkeypatch, version: str, trainer_class: type) -> None:
    trl = types.ModuleType("trl")
    trl.__version__ = version
    trl.GRPOTrainer = trainer_class
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_missing_trl_is_an_import_error_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "trl", None)

    with pytest.raises(ImportError) as excinfo:
        require_trl()

    message = str(excinfo.value)
    assert 'pip install "reservoir[trl]"' in message
    assert PINNED_TRL_VERSION in message


@pytest.mark.parametrize("missing", REQUIRED_TRAINER_ATTRIBUTES)
def test_trainer_without_a_required_attribute_is_rejected(monkeypatch, missing):
    present = tuple(a for a in REQUIRED_TRAINER_ATTRIBUTES if a != missing)
    _install_fake_trl(monkeypatch, PINNED_TRL_VERSION, _fake_trainer_class(*present))

    with pytest.raises(ImportError) as excinfo:
        require_trl()

    message = str(excinfo.value)
    assert missing in message
    assert PINNED_TRL_VERSION in message


def test_untested_version_with_compatible_trainer_warns_and_returns(monkeypatch):
    trainer = _fake_trainer_class(*REQUIRED_TRAINER_ATTRIBUTES)
    _install_fake_trl(monkeypatch, "9.9.9", trainer)

    with pytest.warns(UserWarning, match="9.9.9") as record:
        support = require_trl()

    assert support.grpo_trainer is trainer
    assert support.version == "9.9.9"
    assert len(record) == 1


def test_tested_version_does_not_warn(monkeypatch):
    trainer = _fake_trainer_class(*REQUIRED_TRAINER_ATTRIBUTES)
    _install_fake_trl(monkeypatch, TESTED_TRL_VERSIONS[0], trainer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        support = require_trl()

    assert support.grpo_trainer is trainer


def test_real_trl_passes_the_guard():
    """The installed TRL passes; the version warning fires only for untested versions."""
    trl = pytest.importorskip("trl")
    from trl import GRPOTrainer

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        support = require_trl()

    assert support.grpo_trainer is GRPOTrainer
    ours = [w for w in record if issubclass(w.category, UserWarning) and "reservoir" in str(w.message)]
    if trl.__version__ in TESTED_TRL_VERSIONS:
        assert ours == [], [str(w.message) for w in ours]
    else:
        assert len(ours) == 1


def test_broken_trl_dependency_is_reported_as_such(monkeypatch):
    """A TRL that is installed but fails to import must not be told to install TRL."""
    broken = types.ModuleType("trl")
    monkeypatch.setitem(sys.modules, "trl", broken)
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError) as excinfo:
        require_trl()

    assert "transformers" in str(excinfo.value)


def test_pin_matches_pyproject_extra():
    """The ``trl`` extra in pyproject.toml and the guard must name the same version."""
    import pathlib

    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    pyproject = pathlib.Path(_trl_compat.__file__).parents[3] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml is only present in a source checkout")
    data = tomllib.loads(pyproject.read_text())
    extra = data["project"]["optional-dependencies"]["trl"]
    assert f"trl=={PINNED_TRL_VERSION}" in extra
