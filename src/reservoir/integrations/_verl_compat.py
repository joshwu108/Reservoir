"""Import guard for the verl integration.

verl's trainer internals change often. The adapter in
``reservoir.integrations.verl`` wraps ``RayPPOTrainer``'s worker group and
two of its private checkpoint methods, builds ``DataProto`` objects, and
relies on the GRPO advantage path writing exact zeros for a zero-variance
group. It is tested against one pinned release, so this module is the
single place that checks the installed verl before anything else runs:

- verl missing: ``ImportError`` that names the pinned version and the
  extra that installs it.
- A member the adapter wraps or calls missing (``REQUIRED_TRAINER_ATTRIBUTES``
  on ``RayPPOTrainer``, ``REQUIRED_DATAPROTO_ATTRIBUTES`` on ``DataProto``,
  ``REQUIRED_CORE_ALGOS`` in ``core_algos``): ``ImportError`` naming it.
  The adapter would otherwise fail deep inside a training step, or
  worse, its override would never be called.
- A verl version the adapter has not been tested with but that still has
  every required member: ``UserWarning``. It will probably work, and the
  warning records that nobody has checked.

Keep the three tuples in sync with what the adapter touches; a name added
to the adapter without being added here is a silent dependency.
"""

from __future__ import annotations

import warnings
from typing import Final, NamedTuple

PINNED_VERL_VERSION: Final[str] = "0.7.0"
"""The version ``pip install "reservoir-replay[verl]"`` installs."""

TESTED_VERL_VERSIONS: Final[tuple[str, ...]] = ("0.7.0",)
"""Versions the test suite and the Modal run have been run against."""

REQUIRED_TRAINER_ATTRIBUTES: Final[tuple[str, ...]] = (
    "fit",
    "init_workers",
    "_save_checkpoint",
    "_load_checkpoint",
)
"""Members of ``RayPPOTrainer`` the adapter overrides or calls."""

REQUIRED_DATAPROTO_ATTRIBUTES: Final[tuple[str, ...]] = (
    "from_dict",
    "batch",
    "non_tensor_batch",
    "meta_info",
)
"""Members of ``DataProto`` the adapter reads or calls."""

REQUIRED_CORE_ALGOS: Final[tuple[str, ...]] = ("compute_grpo_outcome_advantage",)
"""The advantage function whose exact zeros define a dead group."""


class VerlSupport(NamedTuple):
    """What ``require_verl`` hands back: the classes the adapter builds on."""

    ray_trainer: type
    data_proto: type
    version: str


def _install_hint() -> str:
    return (
        f'install the supported version with: pip install "reservoir-replay[verl]" '
        f"(pins verl=={PINNED_VERL_VERSION})"
    )


def require_verl() -> VerlSupport:
    """Import verl and verify it exposes what the adapter needs.

    Raises
    ------
    ImportError
        verl is not installed, or a required member is missing.

    Warns
    -----
    UserWarning
        The installed version is not in ``TESTED_VERL_VERSIONS``.
    """
    try:
        import verl
        from verl.protocol import DataProto
        from verl.trainer.ppo import core_algos
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except ModuleNotFoundError as exc:
        if exc.name == "verl" or (exc.name or "").startswith("verl."):
            raise ImportError(f"reservoir.integrations.verl requires verl; " + _install_hint()) from exc
        raise ImportError(f"importing verl failed on a dependency: {exc}") from exc
    except ImportError as exc:
        raise ImportError(f"importing verl failed: {exc}") from exc

    version = str(getattr(verl, "__version__", "unknown"))
    missing = [f"RayPPOTrainer.{n}" for n in REQUIRED_TRAINER_ATTRIBUTES if not hasattr(RayPPOTrainer, n)]
    missing += [f"DataProto.{n}" for n in REQUIRED_DATAPROTO_ATTRIBUTES if not _dataproto_has(DataProto, n)]
    missing += [f"core_algos.{n}" for n in REQUIRED_CORE_ALGOS if not hasattr(core_algos, n)]
    if missing:
        raise ImportError(
            f"the installed verl ({version}) is not supported: it has no {', '.join(missing)}, "
            f"which reservoir.integrations.verl relies on; " + _install_hint()
        )
    if version not in TESTED_VERL_VERSIONS:
        warnings.warn(
            f"reservoir.integrations.verl has been tested with verl {', '.join(TESTED_VERL_VERSIONS)}; "
            f"the installed version is {version}. The required members are present, so it will "
            "probably work, but this combination has not been verified.",
            UserWarning,
            stacklevel=2,
        )
    return VerlSupport(ray_trainer=RayPPOTrainer, data_proto=DataProto, version=version)


def _dataproto_has(cls: type, name: str) -> bool:
    """``DataProto`` is a dataclass: its fields show up in ``__dataclass_fields__``, not as class attributes."""
    return hasattr(cls, name) or name in getattr(cls, "__dataclass_fields__", {})


__all__ = [
    "PINNED_VERL_VERSION",
    "REQUIRED_CORE_ALGOS",
    "REQUIRED_DATAPROTO_ATTRIBUTES",
    "REQUIRED_TRAINER_ATTRIBUTES",
    "TESTED_VERL_VERSIONS",
    "VerlSupport",
    "require_verl",
]
