from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.receipt_mlnet_unified_evaluate import (
    EvaluationInputError,
    main,
    prepare_input_list,
    score_results,
)


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
    status: str | None = None,
    status_class: str = "success",
) -> dict[str, object]:
    slots: dict[str, object] = {
        "amount": {"text": amount_text, "visible_text": amount_visible},
        "time": {"text": time_text, "visible_text": time_visible},
        "payment_method_field": {"text": payment},
        "recipient_field": {"text": recipient},
    }
    if status is not None:
        slots["transfer_status"] = {"text": status, "class_name": status_class}
    return {
        "id": receipt_id,
        "group_id": f"group:{receipt_id}",
        "split": split,
        "source": str(source),
        "slots": slots,
    }


def _result(source: Path, model_sha256: str, **candidates: str | None) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name in ("amount", "time", "payment_method", "recipient", "transfer_status"):
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

    assert summary["accepted"] is False
    assert summary["kind"] == "receipt_mlnet_unified_candidate_evaluation_v1"
    assert summary["evaluation_scope"]["kind"] == "full_split"
    assert summary["evaluation_scope"]["requested_limit"] is None
    assert summary["formal_delivery_gate"] is False
    assert summary["acceptance"]["formal_delivery_gate"] is False
    assert summary["diagnostic_thresholds_passed"] is True
    assert summary["acceptance"]["diagnostic_thresholds_passed"] is True
    assert summary["accuracy_denominators"]["hash_bound"] is False
    assert "unbound_full_split" in summary["warning"]
    assert summary["failures"] == []
    assert summary["artifact_audit"]["all_results_match_model"]
    assert summary["coverage"]["result_coverage"] == 1.0
    assert summary["coverage"]["fully_scored_coverage"] == 1.0
    assert summary["coverage_contract_version"] == 2
    assert summary["coverage"]["coverage_contract_version"] == 2
    assert summary["coverage"]["candidate_coverage_domain"] == "all_expected_receipts"
    assert summary["coverage"]["fully_candidate_covered_receipts"] == 2
    assert summary["coverage"]["all_field_candidate_coverage"] == 1.0
    for metrics in summary["by_field"].values():
        assert metrics["records"] == 2
        assert metrics["reference_records"] == 2
        assert metrics["raw_exact_matches"] == 2
        assert metrics["raw_exact_match"] == 1.0
        assert metrics["candidate_coverage"] == 1.0
        assert metrics["candidate_on_reference_coverage"] == 1.0

    comparisons = [json.loads(line) for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()]
    amounts = {row["id"]: row for row in comparisons if row["field"] == "amount"}
    assert amounts["one"]["reference_text"] == "¥ 1,234.56"
    assert amounts["two"]["reference_text"] == "2.00"
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["accepted"] is False


def test_score_v13_status_uses_visible_raw_text_and_checks_unlabeled_receipt_candidates(
    tmp_path: Path,
) -> None:
    model = tmp_path / "status-text-v13.onnx"
    model.write_bytes(b"unified-v13-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    status_source = tmp_path / "status.png"
    no_status_source = tmp_path / "no-status.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        records,
        [
            _record("status", status_source, status="转账成功"),
            _record("no-status", no_status_source),
        ],
    )
    results = tmp_path / "results"
    manifest: list[dict[str, object]] = []
    for receipt_id, source, status in (
        ("status", status_source, "转账成功"),
        ("no-status", no_status_source, None),
    ):
        result_path = results / f"{receipt_id}.json"
        candidates: dict[str, str | None] = {
            "amount": "¥ 1,234.56",
            "time": "2026-08-06 07:08:09",
            "payment_method": "余额",
            "recipient": "商户甲",
        }
        # Candidate presence is required across all selected receipts, even
        # when one receipt has no status accuracy reference.
        candidates["transfer_status"] = status or "转账成功"
        _write_json(result_path, _result(source, model_sha256, **candidates))
        manifest.append({"source": str(source), "result": str(result_path), "status": "written"})
    _write_json(results / "inference_manifest.json", manifest)

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "evaluation",
        status_floor=0.90,
    )

    assert summary["accepted"] is False
    assert summary["diagnostic_thresholds_passed"] is True
    assert summary["formal_delivery_gate"] is False
    assert summary["floors"]["transfer_status"] == 0.90
    assert summary["acceptance"]["min_status_exact_match"] == 0.90
    assert summary["by_field"]["transfer_status"] == {
        "records": 1,
        "reference_records": 1,
        "denominator": "selected_reference_records",
        "raw_exact_matches": 1,
        "raw_exact_match": 1.0,
        "candidate_records": 1,
        "missing_candidate_records": 0,
        "candidate_coverage": 1.0,
        "candidate_on_reference_records": 1,
        "missing_candidate_on_reference_records": 0,
        "candidate_on_reference_coverage": 1.0,
        "non_success_truth_records": 0,
        "non_success_safety_calibrated": False,
        "non_success_to_success": 0,
    }
    assert summary["all_receipt_candidate_coverage"]["by_field"]["transfer_status"] == {
        "expected_receipts": 2,
        "candidate_records": 2,
        "missing_candidate_records": 0,
        "candidate_coverage": 1.0,
    }
    status_rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation" / "comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["field"] == "transfer_status"
    ]
    assert len(status_rows) == 1
    assert status_rows[0]["reference_text"] == "转账成功"
    assert status_rows[0]["candidate_text"] == "转账成功"
    assert status_rows[0]["raw_exact"] is True


def test_score_v13_status_missing_or_semantic_only_candidate_fails_raw_gate(
    tmp_path: Path,
) -> None:
    model = tmp_path / "status-text-v13.onnx"
    model.write_bytes(b"unified-v13-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    sources = [tmp_path / "one.png", tmp_path / "two.png"]
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        records,
        [
            _record("one", sources[0], status="转账成功"),
            _record("two", sources[1], status="转账成功"),
        ],
    )
    results = tmp_path / "results"
    manifest: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        result_path = results / f"{index}.json"
        _write_json(
            result_path,
            _result(
                source,
                model_sha256,
                amount="¥ 1,234.56",
                time="2026-08-06 07:08:09",
                payment_method="余额",
                recipient="商户甲",
                **({"transfer_status": "成功"} if index == 0 else {}),
            ),
        )
        manifest.append({"source": str(source), "result": str(result_path), "status": "written"})
    _write_json(results / "inference_manifest.json", manifest)

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "evaluation",
        status_floor=0.90,
    )

    assert summary["accepted"] is False
    assert summary["by_field"]["transfer_status"]["raw_exact_match"] == 0.0
    assert summary["by_field"]["transfer_status"]["candidate_coverage"] == 0.5
    assert any(
        "transfer_status: all_receipt_candidate_coverage=0.5000 < 1.0000" in failure
        for failure in summary["failures"]
    )
    assert any(
        "transfer_status: raw_exact_match=0.0000 < 0.9000" in failure
        for failure in summary["failures"]
    )


def test_score_v13_rejects_non_success_promoted_to_success_even_with_zero_floor(
    tmp_path: Path,
) -> None:
    model = tmp_path / "status-text-v13.onnx"
    model.write_bytes(b"unified-v13-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    source = tmp_path / "failed.png"
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        records,
        [_record("failed", source, status="转账失败", status_class="failed")],
    )
    results = tmp_path / "results"
    result_path = results / "failed.json"
    _write_json(
        result_path,
        _result(
            source,
            model_sha256,
            amount="¥ 1,234.56",
            time="2026-08-06 07:08:09",
            payment_method="余额",
            recipient="商户甲",
            transfer_status="转账成功",
        ),
    )
    _write_json(
        results / "inference_manifest.json",
        [{"source": str(source), "result": str(result_path), "status": "written"}],
    )

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "evaluation",
        status_floor=0.0,
    )

    status_metrics = summary["by_field"]["transfer_status"]
    assert summary["accepted"] is False
    assert status_metrics["non_success_truth_records"] == 1
    assert status_metrics["non_success_safety_calibrated"] is True
    assert status_metrics["non_success_to_success"] == 1
    assert summary["acceptance"]["max_non_success_to_success"] == 0
    assert "transfer_status: non_success_to_success=1 > 0" in summary["failures"]


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
    assert any(
        "recipient_field: all_receipt_candidate_coverage=0.0000" in failure
        for failure in summary["failures"]
    )


def test_amount_semantic_is_diagnostic_only_and_preserves_digit_differences(tmp_path: Path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"unified-v12-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    equivalent_source = tmp_path / "equivalent.png"
    different_source = tmp_path / "different.png"
    records = tmp_path / "unified_fields.jsonl"
    equivalent_record = _record(
        "equivalent",
        equivalent_source,
        amount_text="2000.00",
        amount_visible="2000.00",
        time_text="09:10",
        time_visible="09:10",
    )
    equivalent_record["result_json"] = str(tmp_path / "teacher-equivalent.json")
    different_record = _record(
        "different",
        different_source,
        amount_text="300.00",
        amount_visible="300.00",
        time_text="09:10",
        time_visible="09:10",
    )
    _write_jsonl(records, [equivalent_record, different_record])

    results = tmp_path / "results"
    manifest: list[dict[str, object]] = []
    for name, source, amount, ctc, structured, bbox in (
        ("equivalent", equivalent_source, " \t￥2,000.00 \n", "2000.00", "￥2,000.00", [10, 20, 110, 45]),
        ("different", different_source, "￥301.00", "301.00", "￥301.00", [11, 21, 111, 46]),
    ):
        result = _result(
            source,
            model_sha256,
            amount=amount,
            time="09:10",
            payment_method="余额",
            recipient="商户甲",
        )
        amount_field = result["fields"]["amount"]
        amount_field["ctc_candidate"] = ctc
        amount_field["structured_candidate"] = structured
        result["detections"] = [
            {"label": "amount", "score": 0.987654, "bbox_image": bbox},
        ]
        result["geometry"] = {
            "resize_mode": "letterbox",
            "perspective_rectification": "not_applied",
        }
        result_path = results / f"{name}.json"
        _write_json(result_path, result)
        manifest.append({"source": str(source), "result": str(result_path), "status": "written"})
    _write_json(results / "inference_manifest.json", manifest)
    output = tmp_path / "evaluation"

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=output,
        amount_floor=1.0,
    )

    # Both display strings remain strict failures, so this diagnostic cannot
    # weaken the existing formal protection line.
    assert not summary["accepted"]
    assert summary["by_field"]["amount"]["raw_exact_matches"] == 0
    assert any("amount: raw_exact_match=0.0000 < 1.0000" in failure for failure in summary["failures"])
    assert summary["amount_semantic"] == {
        "diagnostic_only": True,
        "affects_acceptance": False,
        "normalization": (
            "strip surrounding whitespace; accept strict v8 CNY display grammar; remove ¥/￥, optional "
            "currency space and valid thousands separators; compare canonical digits with Decimal"
        ),
        "records": 2,
        "reference_parseable_records": 2,
        "candidate_parseable_records": 2,
        "comparable_records": 2,
        "exact_matches": 1,
        "exact_match": 0.5,
    }

    comparisons = [
        json.loads(line)
        for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    amounts = {row["id"]: row for row in comparisons if row["field"] == "amount"}
    assert amounts["equivalent"]["amount_semantic_exact"] is True
    assert amounts["equivalent"]["reference_amount_decimal"] == "2000.00"
    assert amounts["equivalent"]["candidate_amount_decimal"] == "2000.00"
    assert amounts["equivalent"]["ctc_candidate_text"] == "2000.00"
    assert amounts["equivalent"]["structured_candidate_text"] == "￥2,000.00"
    assert amounts["equivalent"]["detection_bbox_image"] == [10.0, 20.0, 110.0, 45.0]
    assert amounts["equivalent"]["detection_score"] == pytest.approx(0.987654)
    assert amounts["equivalent"]["result_geometry"]["perspective_rectification"] == "not_applied"
    assert amounts["equivalent"]["teacher_result_json"] == str(tmp_path / "teacher-equivalent.json")
    assert amounts["different"]["amount_semantic_exact"] is False
    assert amounts["different"]["reference_amount_decimal"] == "300.00"
    assert amounts["different"]["candidate_amount_decimal"] == "301.00"


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


def test_pilot_prepare_and_score_use_deterministic_hash_bound_five_field_selection(
    tmp_path: Path,
) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"unified-v13-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    sources = [tmp_path / f"receipt-{index:02d}.png" for index in range(40)]
    records = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        records,
        [
            _record(
                f"receipt-{index:02d}",
                source,
                recipient=f"商户{index:02d}",
                status="转账成功" if index >= 20 else None,
            )
            for index, source in enumerate(sources)
        ],
    )
    input_list = tmp_path / "pilot-inputs.txt"
    duplicate_input_list = tmp_path / "pilot-inputs-again.txt"

    report = prepare_input_list(
        records_path=records,
        output_path=input_list,
        limit=20,
    )
    duplicate_report = prepare_input_list(
        records_path=records,
        output_path=duplicate_input_list,
        limit=20,
    )

    selected = input_list.read_text(encoding="utf-8").splitlines()
    assert len(selected) == len(set(selected)) == 20
    assert any(int(Path(source).stem.rsplit("-", 1)[1]) >= 20 for source in selected)
    assert input_list.read_bytes() == duplicate_input_list.read_bytes()
    assert report["output_sha256"] == duplicate_report["output_sha256"]
    assert report["selection_order"] == (
        "deterministic_min16_field_quota_then_records_manifest_order"
    )
    assert report["field_quotas"] == {
        "amount": 16,
        "time": 16,
        "payment_method_field": 16,
        "recipient_field": 16,
        "transfer_status": 16,
    }
    assert report["selected_field_reference_counts"]["transfer_status"] == 16

    results = tmp_path / "results"
    manifest: list[dict[str, object]] = []
    source_indexes = {str(source): index for index, source in enumerate(sources)}
    for result_index, selected_source in enumerate(selected):
        source = Path(selected_source)
        source_index = source_indexes[selected_source]
        result_path = results / f"result-{result_index}.json"
        candidates: dict[str, str | None] = {
            "amount": "¥ 1,234.56",
            "time": "2026-08-06 07:08:09",
            "payment_method": "余额",
            "recipient": f"商户{source_index:02d}",
            "transfer_status": "转账成功",
        }
        _write_json(
            result_path,
            _result(source, model_sha256, **candidates),
        )
        manifest.append({"source": str(source), "result": str(result_path), "status": "written"})
    _write_json(results / "inference_manifest.json", manifest)
    output = tmp_path / "evaluation"
    input_sha256 = hashlib.sha256(input_list.read_bytes()).hexdigest()

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
            "--input-list",
            str(input_list),
            "--input-list-sha256",
            input_sha256,
            "--status-floor",
            "0.90",
            "--limit",
            "20",
        ]
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["kind"] == "receipt_mlnet_unified_candidate_partial_pilot_evaluation_v1"
    assert summary["evaluation_scope"] == {
        "kind": "partial_pilot",
        "requested_limit": 20,
        "evaluated_expected_receipts": 20,
        "full_split_expected_receipts": 40,
        "input_list_path": input_list.resolve().as_posix(),
        "input_list_sha256": input_sha256,
        "selection_order": "deterministic_min16_field_quota_then_records_manifest_order",
        "formal_delivery_gate": False,
    }
    assert summary["input_selection"] == {
        "path": input_list.resolve().as_posix(),
        "sha256": input_sha256,
        "hash_bound": True,
        "records": 20,
        "selection_order": "deterministic_min16_field_quota_then_records_manifest_order",
        "field_quotas": report["field_quotas"],
        "field_reference_counts": report["selected_field_reference_counts"],
    }
    assert summary["coverage"]["expected_receipts"] == 20
    assert summary["missing"]["result_receipts"] == 0
    assert summary["failures"] == []
    assert summary["pilot_thresholds_passed"] is True
    assert summary["formal_delivery_gate"] is False
    assert summary["accepted"] is False
    assert summary["acceptance"]["passed"] is False
    assert summary["acceptance"]["pilot_thresholds_passed"] is True
    assert "partial_pilot" in summary["warning"]
    assert "formal_delivery_gate=false" in summary["warning"]

    comparisons = [json.loads(line) for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["source"] for row in comparisons} == set(selected)
    assert summary["by_field"]["transfer_status"]["records"] == 16
    assert len(comparisons) == 96


def test_hash_bound_accuracy_denominator_is_reference_count_but_candidate_gate_is_all_receipts(
    tmp_path: Path,
) -> None:
    model = tmp_path / "v13.onnx"
    model.write_bytes(b"v13-reference-denominator")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    sources = [tmp_path / "with-recipient.png", tmp_path / "without-recipient.png"]
    records = tmp_path / "records.jsonl"
    with_recipient = _record("with", sources[0], recipient="商户甲", status="转账成功")
    without_recipient = _record("without", sources[1], status="转账成功")
    del without_recipient["slots"]["recipient_field"]  # type: ignore[index]
    _write_jsonl(records, [with_recipient, without_recipient])

    results = tmp_path / "results"
    manifest: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        result_path = results / f"{index}.json"
        candidates: dict[str, str | None] = {
            "amount": "¥ 1,234.56",
            "time": "2026-08-06 07:08:09",
            "payment_method": "余额",
            "transfer_status": "转账成功",
        }
        if index == 0:
            candidates["recipient"] = "商户甲"
        _write_json(result_path, _result(source, model_sha256, **candidates))
        manifest.append({"source": str(source), "result": str(result_path), "status": "written"})
    _write_json(results / "inference_manifest.json", manifest)

    input_list = tmp_path / "full-inputs.txt"
    report = prepare_input_list(records_path=records, output_path=input_list)
    input_sha256 = hashlib.sha256(input_list.read_bytes()).hexdigest()
    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "evaluation",
        input_list_path=input_list,
        input_list_sha256=input_sha256,
        status_floor=0.90,
    )

    assert report["selected_field_reference_counts"]["recipient_field"] == 1
    assert summary["input_selection"]["field_reference_counts"]["recipient_field"] == 1
    assert summary["accuracy_denominators"] == {
        "scope": "selected_reference_records",
        "hash_bound": True,
        "source": "input_selection.field_reference_counts",
        "by_field": report["selected_field_reference_counts"],
    }
    assert summary["by_field"]["recipient_field"]["records"] == 1
    assert summary["by_field"]["recipient_field"]["raw_exact_match"] == 1.0
    assert summary["coverage"]["by_field"]["recipient_field"] == {
        "references": 1,
        "candidates": 1,
        "missing": 0,
        "coverage": 1.0,
    }
    assert summary["all_receipt_candidate_coverage"]["expected_receipts"] == 2
    assert summary["all_receipt_candidate_coverage"]["complete_receipts"] == 1
    assert summary["all_receipt_candidate_coverage"]["by_field"]["recipient_field"] == {
        "expected_receipts": 2,
        "candidate_records": 1,
        "missing_candidate_records": 1,
        "candidate_coverage": 0.5,
    }
    assert summary["coverage"]["by_field_all_receipts"]["recipient_field"] == {
        "expected_receipts": 2,
        "candidate_records": 1,
        "missing_candidate_records": 1,
        "candidate_coverage": 0.5,
    }
    assert summary["accepted"] is False
    assert summary["formal_delivery_gate"] is False
    assert any(
        "recipient_field: all_receipt_candidate_coverage=0.5000 < 1.0000" in failure
        for failure in summary["failures"]
    )


def _write_bound_v13_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, list[Path], Path, dict[str, Path]]:
    model = tmp_path / "v13.onnx"
    model.write_bytes(b"v13-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    sources = [tmp_path / f"bound-{index}.png" for index in range(3)]
    records = tmp_path / "bound-records.jsonl"
    _write_jsonl(
        records,
        [
            _record(
                f"bound-{index}",
                source,
                recipient=f"商户{index}",
                status="转账成功",
            )
            for index, source in enumerate(sources)
        ],
    )
    results = tmp_path / "bound-results"
    result_paths: dict[str, Path] = {}
    manifest: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        result_path = results / f"bound-{index}.json"
        _write_json(
            result_path,
            _result(
                source,
                model_sha256,
                amount="¥ 1,234.56",
                time="2026-08-06 07:08:09",
                payment_method="余额",
                recipient=f"商户{index}",
                transfer_status="转账成功",
            ),
        )
        result_paths[str(source)] = result_path
        manifest.append(
            {"source": str(source), "result": str(result_path), "status": "written"}
        )
    manifest_path = results / "inference_manifest.json"
    _write_json(manifest_path, manifest)
    return model, records, sources, results, result_paths


def test_pilot_prepare_rejects_missing_field_or_insufficient_unique_sources(
    tmp_path: Path,
) -> None:
    records_without_status = tmp_path / "without-status.jsonl"
    _write_jsonl(
        records_without_status,
        [_record("one", tmp_path / "one.png"), _record("two", tmp_path / "two.png")],
    )
    with pytest.raises(EvaluationInputError, match="transfer_status"):
        prepare_input_list(
            records_path=records_without_status,
            output_path=tmp_path / "missing-field.txt",
            limit=1,
        )

    records = tmp_path / "too-small.jsonl"
    _write_jsonl(
        records,
        [_record("one", tmp_path / "only.png", status="转账成功")],
    )
    with pytest.raises(EvaluationInputError, match="exceeds 1 unique"):
        prepare_input_list(
            records_path=records,
            output_path=tmp_path / "too-large.txt",
            limit=2,
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("\n", "empty source"),
        ("{first}\n{first}\n", "duplicate source"),
        ("{unknown}\n", "outside the val reference set"),
        ("{second}\n", "does not match the deterministic"),
    ],
)
def test_pilot_score_rejects_invalid_or_unbound_explicit_input_list(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    model, records, sources, results, _ = _write_bound_v13_fixture(tmp_path)
    input_list = tmp_path / "invalid-inputs.txt"
    input_list.write_text(
        contents.format(first=sources[0], second=sources[1], unknown=tmp_path / "unknown.png"),
        encoding="utf-8",
    )
    input_sha256 = hashlib.sha256(input_list.read_bytes()).hexdigest()

    with pytest.raises(EvaluationInputError, match=message):
        score_results(
            records_path=records,
            results_root=results,
            model_path=model,
            output_dir=tmp_path / "invalid-evaluation",
            input_list_path=input_list,
            input_list_sha256=input_sha256,
            status_floor=0.90,
            limit=1,
        )


def test_pilot_score_requires_paired_input_list_hash_and_rejects_wrong_hash(
    tmp_path: Path,
) -> None:
    model, records, _, results, _ = _write_bound_v13_fixture(tmp_path)
    input_list = tmp_path / "pilot.txt"
    prepare_input_list(records_path=records, output_path=input_list, limit=1)
    input_sha256 = hashlib.sha256(input_list.read_bytes()).hexdigest()
    common = {
        "records_path": records,
        "results_root": results,
        "model_path": model,
        "output_dir": tmp_path / "evaluation",
        "status_floor": 0.90,
        "limit": 1,
    }

    with pytest.raises(EvaluationInputError, match="provided together"):
        score_results(**common, input_list_path=input_list)
    with pytest.raises(EvaluationInputError, match="provided together"):
        score_results(**common, input_list_sha256=input_sha256)
    with pytest.raises(EvaluationInputError, match="hash-bound explicit input list"):
        score_results(**common)
    with pytest.raises(EvaluationInputError, match="SHA-256 mismatch"):
        score_results(
            **common,
            input_list_path=input_list,
            input_list_sha256="0" * 64,
        )


@pytest.mark.parametrize("manifest_kind", ["empty", "duplicate", "outside"])
def test_bound_score_rejects_empty_duplicate_or_outside_inference_manifest(
    tmp_path: Path,
    manifest_kind: str,
) -> None:
    model, records, sources, results, result_paths = _write_bound_v13_fixture(tmp_path)
    input_list = tmp_path / "pilot.txt"
    prepare_input_list(records_path=records, output_path=input_list, limit=1)
    input_sha256 = hashlib.sha256(input_list.read_bytes()).hexdigest()
    if manifest_kind == "empty":
        manifest: list[dict[str, object]] = []
        message = "is empty for the bound input list"
    elif manifest_kind == "duplicate":
        manifest = [
            {"source": str(sources[0]), "status": "failed"},
            {"source": str(sources[0]), "status": "failed"},
        ]
        message = "contains duplicate source"
    else:
        manifest = [
            {
                "source": str(sources[1]),
                "result": str(result_paths[str(sources[1])]),
                "status": "written",
            }
        ]
        message = "outside the hash-bound explicit input list"
    _write_json(results / "inference_manifest.json", manifest)

    with pytest.raises(EvaluationInputError, match=message):
        score_results(
            records_path=records,
            results_root=results,
            model_path=model,
            output_dir=tmp_path / "manifest-evaluation",
            input_list_path=input_list,
            input_list_sha256=input_sha256,
            status_floor=0.90,
            limit=1,
        )


def test_formal_bound_input_list_must_be_complete_and_remains_formal(
    tmp_path: Path,
) -> None:
    model, records, sources, results, _ = _write_bound_v13_fixture(tmp_path)
    full_input_list = tmp_path / "full-inputs.txt"
    report = prepare_input_list(records_path=records, output_path=full_input_list)
    full_sha256 = hashlib.sha256(full_input_list.read_bytes()).hexdigest()

    summary = score_results(
        records_path=records,
        results_root=results,
        model_path=model,
        output_dir=tmp_path / "formal-evaluation",
        input_list_path=full_input_list,
        input_list_sha256=full_sha256,
        status_floor=0.90,
    )

    assert report["selection_order"] == "first_unique_source_in_records_manifest_order"
    assert summary["accepted"] is True
    assert summary["formal_delivery_gate"] is True
    assert summary["evaluation_scope"]["kind"] == "full_split"
    assert summary["evaluation_scope"]["input_list_sha256"] == full_sha256
    assert summary["input_selection"]["records"] == 3

    subset = tmp_path / "formal-subset.txt"
    subset.write_text(str(sources[0]) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="explicit subsets are forbidden"):
        score_results(
            records_path=records,
            results_root=results,
            model_path=model,
            output_dir=tmp_path / "formal-subset-evaluation",
            input_list_path=subset,
            input_list_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
            status_floor=0.90,
        )
