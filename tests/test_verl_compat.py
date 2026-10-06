"""Tests for the verl import guard in ``reservoir.integrations._verl_compat``.

A missing verl is an ``ImportError`` that says how to install the pinned
version, a verl whose ``RayPPOTrainer``, ``DataProto`` or ``core_algos`` has
lost a member the adapter relies on is an ``ImportError`` naming it, and an
untested verl version is a warning rather than a failure. Fake ``verl``
modules are installed in ``sys.modules`` so the tests run without verl.
"""

from __future__ import annotations

import dataclasses
import sys
import types
import warnings

import pytest

from reservoir.integrations import _verl_compat
from reservoir.integrations._verl_compat import (
    PINNED_VERL_VERSION,
    REQUIRED_CORE_ALGOS,
    REQUIRED_DATAPROTO_ATTRIBUTES,
    REQUIRED_TRAINER_ATTRIBUTES,
    TESTED_VERL_VERSIONS,
    require_verl,
)


def _trainer_class(*attributes: str) -> type:
    return type("RayPPOTrainer", (), {name: (lambda self, *a, **k: None) for name in attributes})


def _dataproto_class(*attributes: str) -> type:
    fields = [(name, object, dataclasses.field(default=None)) for name in attributes if name != "from_dict"]
    cls = dataclasses.make_dataclass("DataProto", fields)
    if "from_dict" in attributes:
        cls.from_dict = classmethod(lambda cls, **k: cls())
    return cls


def _install_fake_verl(monkeypatch, version: str, trainer: type, dataproto: type, core_algos_names=REQUIRED_CORE_ALGOS) -> None:
    verl = types.ModuleType("verl")
    verl.__version__ = version
    protocol = types.ModuleType("verl.protocol")
    protocol.DataProto = dataproto
    trainer_pkg = types.ModuleType("verl.trainer")
    ppo = types.ModuleType("verl.trainer.ppo")
    core_algos = types.ModuleType("verl.trainer.ppo.core_algos")
    for name in core_algos_names:
        setattr(core_algos, name, lambda *a, **k: None)
    ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")
    ray_trainer.RayPPOTrainer = trainer
    for name, module in {
        "verl": verl, "verl.protocol": protocol, "verl.trainer": trainer_pkg, "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.core_algos": core_algos, "verl.trainer.ppo.ray_trainer": ray_trainer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    verl.protocol = protocol
    verl.trainer = trainer_pkg
    trainer_pkg.ppo = ppo
    ppo.core_algos = core_algos
    ppo.ray_trainer = ray_trainer


def _complete() -> tuple[type, type]:
    return _trainer_class(*REQUIRED_TRAINER_ATTRIBUTES), _dataproto_class(*REQUIRED_DATAPROTO_ATTRIBUTES)


def test_missing_verl_is_an_import_error_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "verl", None)
    with pytest.raises(ImportError) as excinfo:
        require_verl()
    assert 'pip install "reservoir-replay[verl]"' in str(excinfo.value)
    assert PINNED_VERL_VERSION in str(excinfo.value)


@pytest.mark.parametrize("missing", REQUIRED_TRAINER_ATTRIBUTES)
def test_trainer_without_a_required_attribute_is_rejected(monkeypatch, missing):
    trainer = _trainer_class(*(a for a in REQUIRED_TRAINER_ATTRIBUTES if a != missing))
    _install_fake_verl(monkeypatch, PINNED_VERL_VERSION, trainer, _dataproto_class(*REQUIRED_DATAPROTO_ATTRIBUTES))
    with pytest.raises(ImportError, match=f"RayPPOTrainer.{missing}"):
        require_verl()


@pytest.mark.parametrize("missing", REQUIRED_DATAPROTO_ATTRIBUTES)
def test_dataproto_without_a_required_member_is_rejected(monkeypatch, missing):
    dataproto = _dataproto_class(*(a for a in REQUIRED_DATAPROTO_ATTRIBUTES if a != missing))
    _install_fake_verl(monkeypatch, PINNED_VERL_VERSION, _trainer_class(*REQUIRED_TRAINER_ATTRIBUTES), dataproto)
    with pytest.raises(ImportError, match=f"DataProto.{missing}"):
        require_verl()


def test_core_algos_without_the_grpo_advantage_is_rejected(monkeypatch):
    trainer, dataproto = _complete()
    _install_fake_verl(monkeypatch, PINNED_VERL_VERSION, trainer, dataproto, core_algos_names=())
    with pytest.raises(ImportError, match="core_algos.compute_grpo_outcome_advantage"):
        require_verl()


def test_untested_version_with_compatible_members_warns_and_returns(monkeypatch):
    trainer, dataproto = _complete()
    _install_fake_verl(monkeypatch, "9.9.9", trainer, dataproto)
    with pytest.warns(UserWarning, match="9.9.9") as record:
        support = require_verl()
    assert support.ray_trainer is trainer and support.data_proto is dataproto and support.version == "9.9.9"
    assert len(record) == 1


def test_tested_version_does_not_warn(monkeypatch):
    trainer, dataproto = _complete()
    _install_fake_verl(monkeypatch, TESTED_VERL_VERSIONS[0], trainer, dataproto)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert require_verl().ray_trainer is trainer


def test_broken_verl_dependency_is_reported_as_such(monkeypatch):
    broken = types.ModuleType("verl")
    monkeypatch.setitem(sys.modules, "verl", broken)
    monkeypatch.setitem(sys.modules, "verl.protocol", None)
    with pytest.raises(ImportError) as excinfo:
        require_verl()
    assert "reservoir-replay[verl]" in str(excinfo.value)


def test_trainer_class_is_built_lazily_through_the_guard(monkeypatch):
    import reservoir.integrations.verl as adapter

    trainer, dataproto = _complete()
    trainer.__init__ = lambda self, *a, **k: None
    _install_fake_verl(monkeypatch, TESTED_VERL_VERSIONS[0], trainer, dataproto)
    adapter.build_trainer_class.cache_clear()
    cls = adapter.ReservoirRayPPOTrainer
    assert issubclass(cls, trainer) and issubclass(cls, adapter.ReservoirReplayMixin)
    assert "ReservoirRayPPOTrainer" in dir(adapter)
    with pytest.raises(TypeError, match="replay_buffer"):
        cls(replay_buffer=object())
    instance = cls(replay_buffer=adapter.ReservoirReplay(capacity=4))
    assert instance.replay_buffer.buffer.capacity == 4
    with pytest.raises(AttributeError):
        adapter.no_such_name
    adapter.build_trainer_class.cache_clear()


def test_real_verl_passes_the_guard():
    verl = pytest.importorskip("verl")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        support = require_verl()
    ours = [w for w in record if issubclass(w.category, UserWarning) and "reservoir" in str(w.message)]
    assert support.version == verl.__version__
    assert ours == [] if verl.__version__ in TESTED_VERL_VERSIONS else len(ours) == 1


def test_pin_matches_pyproject_extra():
    import pathlib

    tomllib = pytest.importorskip("tomllib")
    pyproject = pathlib.Path(_verl_compat.__file__).parents[3] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml is only present in a source checkout")
    extra = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]["verl"]
    assert f"verl=={PINNED_VERL_VERSION}" in extra
