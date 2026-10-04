"""The crash campaign's verdict logic: a row passes only if the cut fired and recovery is clean."""

from __future__ import annotations

import signal
from types import SimpleNamespace

from campaigns.crash import _verdict, _wait_for_cut


class FakeProcess:
    def __init__(self, exitcode, alive_after_join=False):
        self.exitcode = exitcode
        self._alive = alive_after_join
        self.pid = 1
        self.killed = False

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self._alive


def test_cut_fired_only_on_sigkill_exit():
    assert _wait_for_cut(FakeProcess(-signal.SIGKILL)) is True
    assert _wait_for_cut(FakeProcess(0)) is False
    assert _wait_for_cut(FakeProcess(1)) is False
    assert _wait_for_cut(FakeProcess(-signal.SIGTERM)) is False


def test_timeout_is_not_a_fired_cut(monkeypatch):
    killed = []
    monkeypatch.setattr("campaigns.crash.os.kill", lambda pid, sig: killed.append((pid, sig)))
    proc = FakeProcess(None, alive_after_join=True)
    assert _wait_for_cut(proc, timeout=0) is False
    assert killed == [(1, signal.SIGKILL)]


def test_verdict_requires_fired_cut_and_exact_state():
    pre, post = {"a": 1}, {"a": 2}
    assert _verdict("insert", "c", 0, pre, pre, post, cut_fired=True)["passed"]
    assert _verdict("insert", "c", 0, post, pre, post, cut_fired=True)["passed"]
    completed = _verdict("insert", "c", 0, post, pre, post, cut_fired=False)
    assert not completed["passed"] and not completed["torn"]
    torn = _verdict("insert", "c", 0, {"a": 3}, pre, post, cut_fired=True)
    assert torn["torn"] and not torn["passed"]
