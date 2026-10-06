"""Import guard for the TRL integration.

TRL's trainer internals change often. The adapter in
``reservoir.integrations.trl`` relies on a few private members of
``trl.GRPOTrainer`` and is tested against one pinned release, so this
module is the single place that checks the installed TRL before anything
else runs:

- TRL (or transformers) missing: ``ImportError`` that names the pinned
  version and the extra that installs it.
- ``GRPOTrainer`` lacking a member the adapter calls: ``ImportError``
  naming the member. The adapter would otherwise fail later with an
  ``AttributeError`` deep inside a training step, or worse, its override
  would never be called.
- A TRL version the adapter has not been tested with but that still has
  every required member: ``UserWarning``. It will probably work, and the
  warning records that nobody has checked.

Keep ``REQUIRED_TRAINER_ATTRIBUTES`` in sync with what the adapter calls
on the trainer; a name added to the adapter without being added here is
a silent dependency.
"""

from __future__ import annotations

import warnings
from typing import Final, NamedTuple

PINNED_TRL_VERSION: Final[str] = "1.13.0"
"""The version ``pip install "reservoir-replay[trl]"`` installs."""

TESTED_TRL_VERSIONS: Final[tuple[str, ...]] = ("1.13.0",)
"""Versions the test suite has been run against."""

REQUIRED_TRAINER_ATTRIBUTES: Final[tuple[str, ...]] = (
    "_generate_and_score_completions",
    "_prepare_inputs",
    "_get_per_token_logps_and_entropies",
    "_calculate_rewards",   # wrapped to capture per-reward-function values for provenance
)
"""Members of ``trl.GRPOTrainer`` the adapter overrides or calls."""


class TrlSupport(NamedTuple):
    """What ``require_trl`` hands back: the classes the adapter builds on."""

    grpo_trainer: type
    trainer_callback: type
    version: str


def _install_hint() -> str:
    return (
        f'install the supported version with: pip install "reservoir-replay[trl]" '
        f"(pins trl=={PINNED_TRL_VERSION})"
    )


def require_trl() -> TrlSupport:
    """Import TRL and verify it exposes what the adapter needs.

    Returns the ``GRPOTrainer`` and ``TrainerCallback`` classes together
    with the installed version string.

    Raises
    ------
    ImportError
        TRL or transformers is not installed, or ``GRPOTrainer`` is
        missing a member listed in ``REQUIRED_TRAINER_ATTRIBUTES``.

    Warns
    -----
    UserWarning
        The installed version is not in ``TESTED_TRL_VERSIONS``.
    """
    try:
        import trl
        from transformers import TrainerCallback
        from trl import GRPOTrainer
    except ModuleNotFoundError as exc:
        if exc.name in ("trl", "transformers"):
            raise ImportError(
                f"reservoir.integrations.trl requires {exc.name}; " + _install_hint()
            ) from exc
        raise ImportError(f"importing TRL failed on a dependency: {exc}") from exc
    except ImportError as exc:
        raise ImportError(f"importing TRL failed: {exc}") from exc

    missing = [name for name in REQUIRED_TRAINER_ATTRIBUTES if not hasattr(GRPOTrainer, name)]
    version = str(getattr(trl, "__version__", "unknown"))
    if missing:
        raise ImportError(
            f"the installed TRL ({version}) is not supported: trl.GRPOTrainer has no "
            f"{', '.join(missing)}, which reservoir.integrations.trl relies on; "
            + _install_hint()
        )
    if version not in TESTED_TRL_VERSIONS:
        warnings.warn(
            f"reservoir.integrations.trl has been tested with TRL "
            f"{', '.join(TESTED_TRL_VERSIONS)}; the installed version is {version}. "
            "The required trainer members are present, so it will probably work, but "
            "this combination has not been verified.",
            UserWarning,
            stacklevel=2,
        )
    return TrlSupport(grpo_trainer=GRPOTrainer, trainer_callback=TrainerCallback, version=version)


__all__ = [
    "PINNED_TRL_VERSION",
    "REQUIRED_TRAINER_ATTRIBUTES",
    "TESTED_TRL_VERSIONS",
    "TrlSupport",
    "require_trl",
]
