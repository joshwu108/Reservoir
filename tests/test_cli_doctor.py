from __future__ import annotations

from reservoir.cli import doctor_rows, main


def test_doctor_round_trip_and_checker(capsys):
    rows = {name: (status, detail) for name, status, detail in doctor_rows()}
    assert rows["durable round trip"][0] == "OK"
    assert rows["independent checker"][0] == "OK"
    assert "attest.jsonl" in rows["trainer logging"][1]
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "trl version" in output
    assert "verl version" in output
    assert "GPU visible" in output
