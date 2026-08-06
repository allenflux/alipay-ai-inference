from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.receipt_mlnet_unified_evaluate import main, prepare_input_list, score_results


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _record(
    receipt_id: str,
    source: Path,
    *,
    split: str = "val",
    amount_text: str = "1234.56",
    amount_visible: str = "¥ 1,234.56",
    time_text: str = "07:08:09",
    time_visible: str = "2026-08-06 07:08:09",
    payment: str = "余额",
    recipient: str = "商户甲",
) -> dict[str, object]:
    return {
        "id": receipt_id,
        "group_id": f"group:{receipt_id}",
        "split": split,
        "source": str(source),
        "slots": {
            "amount": {"text": amount_text, "visible_text": amount_visible},
            "time": {"text": time_text, "visible_text": time_visible},
            "payment_method_field": {"text": payment},
            "recipient_field": {"text": recipient},
        },
    }


def _result(source: Path, model_sha256: str, **candidates: str | None) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name in ("amount", "time", "payment_method", "recipient"):
        if name in candidates:
            fields[name] = {"candidate": candidates[name]}
    return {
        "source": str(source),
        "fields": fields,
        "model_contracts": {"unified_ocr_model_sha256": model_sha256},
    }


def test_prepare_writes_unique_val_sources_as_utf8_without_bom(tmp_path: Path) -> None:
    first = tmp_path / "回单甲.png"
    second = tmp_path / "回单乙.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        records,
        [
            _record("train", tmp_path / "train.png", split="train"),
            _record("val-a", first),
            _record("val-a-copy", first),
            _record("val-b", second),
        ],
    )
    output = tmp_path / "val-inputs.txt"

    report = prepare_input_list(records_path=records, output_path=output)

    assert report["records"] == 3
    assert report["unique_sources"] == 2
    assert output.read_bytes().startswith(str(first).encode("utf-8"))
    assert not output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert output.read_text(encoding="utf-8").splitlines() == [str(first), str(second)]


def test_score_matches_v12_references_and_accepts_uniform_model(tmp_path: Path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"unified-v12-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    first_source = tmp_path / "one.png"
    second_source = tmp_path / "two.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        records,
        [
            _record("one", first_source),
            # Invalid visible amount must fall back to the v12 canonical text.
            _record(
                "two",
                second_source,
                amount_text="2.00",
                amount_visible="RMB 2.00",
                time_text="09:10",
                time_visible="09:10",
                payment="招商银行(1234)",
                recipient="商户乙",
            ),
        ],
    )
    results = tmp_path / "results"
    first_result = results / "one.json"
    second_result = results / "nested" / "two.json"
    _write_json(
        first_result,
        _result(
            first_source,
            model_sha256,
            amount="¥ 1,234.56",
            time="2026-08-06 07:08:09",
            payment_method="余额",
            recipient="商户甲",
        ),
    )
    _write_json(
        second_result,
        _result(
            second_source,
            model_sha256,
            amount="2.00",
            time="09:10",
            payment_method="招商银行(1234)",
            recipient="商户乙",
        ),
    )
    _write_json(
        results / "inference_manifest.json",
        [
            {"source": str(first_source), "result": str(first_result), "status": "written"},
            {"source": str(second_source), "result": str(second_result), "status": "skipped_existing"},
        ],
    )
    output = tmp_path / "evaluation"

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=output,
    )

    assert summary["accepted"]
    assert summary["failures"] == []
    assert summary["artifact_audit"]["all_results_match_model"]
    assert summary["coverage"]["result_coverage"] == 1.0
    assert summary["coverage"]["fully_scored_coverage"] == 1.0
    for metrics in summary["by_field"].values():
        assert metrics["records"] == 2
        assert metrics["raw_exact_matches"] == 2
        assert metrics["raw_exact_match"] == 1.0
        assert metrics["candidate_coverage"] == 1.0

    comparisons = [json.loads(line) for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()]
    amounts = {row["id"]: row for row in comparisons if row["field"] == "amount"}
    assert amounts["one"]["reference_text"] == "¥ 1,234.56"
    assert amounts["two"]["reference_text"] == "2.00"
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["accepted"] is True


def test_score_counts_missing_result_and_candidate_as_errors(tmp_path: Path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"unified-v12-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    first_source = tmp_path / "one.png"
    second_source = tmp_path / "two.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(records, [_record("one", first_source), _record("two", second_source)])
    results = tmp_path / "results"
    first_result = results / "one.json"
    _write_json(
        first_result,
        _result(
            first_source,
            model_sha256,
            amount="WRONG",
            time="2026-08-06 07:08:09",
            payment_method="余额",
            # Deliberately omit fields.recipient.
        ),
    )
    _write_json(
        results / "inference_manifest.json",
        [{"source": str(first_source), "result": str(first_result), "status": "written"}],
    )

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "evaluation",
        amount_floor=0.0,
        time_floor=0.0,
        payment_floor=0.0,
        recipient_floor=0.0,
    )

    assert not summary["accepted"]
    assert summary["missing"]["result_receipts"] == 1
    assert summary["by_field"]["amount"]["raw_exact_matches"] == 0
    assert summary["by_field"]["amount"]["records"] == 2
    assert summary["by_field"]["recipient_field"]["missing_candidate_records"] == 2
    assert summary["coverage"]["result_coverage"] == 0.5
    assert any("result_coverage=0.5000" in failure for failure in summary["failures"])
    assert any("recipient_field: candidate_coverage=0.0000" in failure for failure in summary["failures"])


def test_score_rejects_source_mismatch_and_wrong_model_sha(tmp_path: Path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"expected-model")
    source = tmp_path / "one.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(records, [_record("one", source)])
    results = tmp_path / "results"
    wrong_source_result = results / "wrong-source.json"
    wrong_hash_result = results / "wrong-hash.json"
    _write_json(
        wrong_source_result,
        _result(
            tmp_path / "different.png",
            "0" * 64,
            amount="¥ 1,234.56",
            time="2026-08-06 07:08:09",
            payment_method="余额",
            recipient="商户甲",
        ),
    )
    extra_source = tmp_path / "extra.png"
    _write_json(
        wrong_hash_result,
        _result(
            extra_source,
            "1" * 64,
            amount="1.00",
            time="01:02",
            payment_method="余额",
            recipient="商户丙",
        ),
    )
    _write_json(
        results / "inference_manifest.json",
        [
            {"source": str(source), "result": str(wrong_source_result), "status": "written"},
            {"source": str(extra_source), "result": str(wrong_hash_result), "status": "written"},
        ],
    )

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "evaluation",
        amount_floor=0.0,
        time_floor=0.0,
        payment_floor=0.0,
        recipient_floor=0.0,
    )

    assert not summary["accepted"]
    assert not summary["artifact_audit"]["all_results_match_model"]
    assert len(summary["artifact_audit"]["source_mismatches"]) == 1
    assert set(summary["artifact_audit"]["mismatched_unified_model_sha256_sources"]) == {
        str(source),
        str(extra_source),
    }
    assert any("source mismatches" in failure for failure in summary["failures"])
    assert any("different unified model" in failure for failure in summary["failures"])
    assert any("outside the val reference set" in failure for failure in summary["failures"])


def test_cli_score_returns_one_after_writing_failed_report(tmp_path: Path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"model")
    source = tmp_path / "one.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(records, [_record("one", source)])
    results = tmp_path / "results"
    _write_json(results / "inference_manifest.json", [])
    output = tmp_path / "evaluation"

    exit_code = main(
        [
            "score",
            "--records",
            str(records),
            "--results",
            str(results),
            "--model",
            str(model),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert (output / "summary.json").is_file()
