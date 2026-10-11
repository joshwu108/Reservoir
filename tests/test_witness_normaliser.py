"""The batch witness digest covers the loss normaliser.

TRL's ``num_items_in_batch`` and verl's ``meta_info["global_token_num"]``
scale every row's gradient, so a wrong value must change the digest the
witness commits to.
"""

from __future__ import annotations

import torch

from reservoir.attest import AttestationLog
from reservoir.integrations import _verl_rows
from reservoir.integrations.trl import tensor_digest as trl_digest
from tests.test_trl_rows import make_output
from tests.test_verl_replay import FakeTrainer, live_batch, mixed_batch, replay
from tests.test_verl_rows import make_batch


def test_trl_digest_changes_with_num_items_in_batch():
    out = make_output(prompts=[[1], [1]], completions=[[2, 3], [4]], advantages=[1.0, -1.0])
    before = trl_digest(out)
    assert trl_digest({**out, "num_items_in_batch": out["num_items_in_batch"] + 1}) != before
    assert trl_digest({**out, "num_items_in_batch": out["num_items_in_batch"].clone()}) == before


def test_trl_digest_accepts_a_plain_int_normaliser_equal_to_the_tensor():
    out = make_output(prompts=[[1], [1]], completions=[[2, 3], [4]], advantages=[1.0, -1.0])
    assert trl_digest({**out, "num_items_in_batch": int(out["num_items_in_batch"])}) == trl_digest(out)


def test_verl_digest_changes_with_global_token_num():
    batch, _ = make_batch(prompts=[[1], [1]], responses=[[2, 3], [4]], advantages=[1.0, -1.0])
    tokens = _verl_rows.global_token_num(batch["attention_mask"])
    before = _verl_rows.tensor_digest(batch, token_nums=tokens)
    assert _verl_rows.tensor_digest(batch, token_nums=[tokens[0] + 1, tokens[1]]) != before
    assert _verl_rows.tensor_digest(batch, token_nums=list(tokens)) == before


def test_verl_witness_digest_is_over_the_returned_tensors_and_global_token_num():
    log = AttestationLog()
    r = replay(attest=log)
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    out = trainer.step(mixed_batch(), step=2)

    (witness,) = [x for x in log.records if x["op"] == "batch"]
    expected = _verl_rows.tensor_digest(out.batch, token_nums=out.meta_info["global_token_num"])
    assert witness["tensor_digest"] == expected
    assert out.meta_info["global_token_num"] == torch.as_tensor(out.batch["attention_mask"]).sum(dim=-1).tolist()
