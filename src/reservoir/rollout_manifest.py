"""
reservoir.rollout_manifest — The data behind each insert record's content digest.

An attestation log commits to every stored rollout through the
``content_digest`` on its ``insert`` record (``rollout.content_digest_of``).
The manifest is the opening of that commitment: one JSON line per insert
with the prompt id, the completion tokens, the reward and the source tag,
keyed by the same ``(op_counter, index)`` as the log record. Given both
files, ``python -m checker.verify <log> --manifest <manifest>`` recomputes
every digest and confirms the log commits to exactly these examples;
``python -m checker.transcript`` then reports which of them were sampled,
how often and from which source.

The manifest is not hash-chained. The log already commits to every line
through the digests, so a second chain would add nothing a verifier can
use. A team may publish the log alone (commitments only) or the log with
the manifest (openings too).

File discipline is the attestation file's: a fresh file per run (``"x"``
mode refuses an existing one), one canonical JSON line per record,
flushed as written. ``DurableRolloutBuffer`` passes ``overwrite=True`` and
rewrites the file from recovered state on reopen, through ``restore``. A
``ManifestWriter()`` with no path keeps records in memory only, the way an
``AttestationLog`` does for the log; the durable buffer's probe buffers use
that to validate a snapshot without touching any file.

``restore`` rewrites the file in place (truncate, then write). That is not
atomic on its own; it is safe because the durable buffer's committed state
is the source of truth and the file is rebuilt from it on every reopen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Final, Iterable, Optional, Union

from reservoir.rollout import content_digest_of

MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "op_counter", "index", "content_digest", "prompt_id", "source",
    "tokens", "reward_hex", "entry_version",
)
"""Every manifest line has exactly these keys; ``source`` is ``null`` when unset."""


def manifest_record(
    *,
    op_counter: int,
    index: int,
    prompt_id: str,
    source: Optional[str],
    tokens: Iterable[int],
    reward: float,
    entry_version: int,
) -> dict:
    """Build one manifest line. The digest is computed here from the same inputs."""
    tokens = [int(t) for t in tokens]
    return {
        "op_counter": int(op_counter),
        "index": int(index),
        "content_digest": content_digest_of(prompt_id, tokens, reward),
        "prompt_id": prompt_id,
        "source": source,
        "tokens": tokens,
        "reward_hex": float(reward).hex(),
        "entry_version": int(entry_version),
    }


def validate_manifest_records(records: object) -> list[dict]:
    """Check recovered manifest records field by field and return copies.

    Every record must have exactly ``MANIFEST_KEYS`` with well-typed
    values, and its ``content_digest`` must equal the digest recomputed
    from its own ``prompt_id``, ``tokens`` and ``reward_hex``. A snapshot
    is a file on disk and may have been edited; this is the boundary
    where it is trusted again. Raises ``ValueError`` naming the record.
    """
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    out: list[dict] = []
    for i, record in enumerate(records):
        where = f"manifest record {i}"
        if not isinstance(record, dict) or set(record) != set(MANIFEST_KEYS):
            raise ValueError(f"{where} does not have the manifest keys")
        for name in ("op_counter", "index", "entry_version"):
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{where}: {name} must be a non-negative int, got {value!r}")
        if record["source"] is not None and not isinstance(record["source"], str):
            raise ValueError(f"{where}: source must be a str or null")
        if not isinstance(record["reward_hex"], str):
            raise ValueError(f"{where}: reward_hex must be a str")
        try:
            reward = float.fromhex(record["reward_hex"])
            digest = content_digest_of(record["prompt_id"], record["tokens"], reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if reward.hex() != record["reward_hex"]:
            raise ValueError(f"{where}: reward_hex must be the canonical float.hex() spelling")
        if digest != record["content_digest"]:
            raise ValueError(f"{where}: content_digest does not match its prompt, tokens and reward")
        out.append(_copy(record))
    return out


def _canonical_line(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


def _copy(record: dict) -> dict:
    """A copy that shares nothing mutable with the caller (the token list is the only container)."""
    return {**record, "tokens": list(record["tokens"])}


class ManifestWriter:
    """Collects manifest records in memory and, with a path, mirrors them to a file.

    Parameters
    ----------
    path : str | Path | None
        File to write. ``None`` keeps records in memory only (the durable
        buffer's probe buffers use this). An existing file is refused
        unless ``overwrite`` is true.
    """

    def __init__(self, path: Union[str, Path, None] = None, overwrite: bool = False) -> None:
        self._records: list[dict] = []
        self._file: Optional[IO[str]] = None
        if path is not None:
            self._file = open(Path(path), "w" if overwrite else "x", encoding="utf-8")

    @property
    def records(self) -> list[dict]:
        """All manifest records written so far, in order (copies; mutating them changes nothing)."""
        return [_copy(r) for r in self._records]

    def write(self, record: dict) -> dict:
        """Append one record (as built by ``manifest_record``) and mirror it.

        The file is written first; the in-memory copy is added only once
        the write succeeded, so a failed write cannot leave memory ahead
        of the file.
        """
        if self._file is not None:
            self._file.write(_canonical_line(record))
            self._file.flush()
        self._records.append(_copy(record))
        return record

    def restore(self, records: list[dict]) -> None:
        """Replace all records with validated recovered ones and rewrite the file."""
        validated = validate_manifest_records(records)
        self._records = validated
        if self._file is not None:
            self._file.seek(0)
            self._file.truncate()
            for record in self._records:
                self._file.write(_canonical_line(record))
            self._file.flush()

    def close(self) -> None:
        """Close the file, if any. Safe to call more than once."""
        if self._file is not None:
            self._file.close()
            self._file = None
