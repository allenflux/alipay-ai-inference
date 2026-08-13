from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "otherimages-white-prefix-materialize.py"
SPEC = importlib.util.spec_from_file_location("otherimages_white_prefix_materialize", SCRIPT)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


@pytest.fixture
def small_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert materializer.SOURCE_IMAGE_COUNT == 10_000
    assert materializer.PILOT_IMAGE_COUNT == 1_000
    monkeypatch.setattr(materializer, "SOURCE_IMAGE_COUNT", 4)
    monkeypatch.setattr(materializer, "PILOT_IMAGE_COUNT", 2)


def _build_publication(
    tmp_path: Path,
    *,
    mutate_contract=None,
    reorder_manifest: bool = False,
    timestamp: str | None = None,
) -> tuple[Path, Path, list[dict[str, object]], bytes, str]:
    source = tmp_path / "raw" / "white-4-fixture"
    source.mkdir(parents=True)
    if timestamp is None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    salt = "fixed-test-salt"
    candidates = [
        ("upper.PNG", b"image-upper"),
        ("nested/two.jpg", b"image-two"),
        ("three.webp", b"image-three"),
        ("nested/four.bmp", b"image-four"),
    ]
    keyed: list[tuple[str, str, bytes]] = []
    for relative, payload in candidates:
        normalized = relative.lower()
        key = _sha(f"{salt}\n{normalized}".encode())
        keyed.append((key, relative, payload))
    keyed.sort(key=lambda item: (item[0], item[1].lower()))
    if reorder_manifest:
        keyed[0], keyed[1] = keyed[1], keyed[0]

    rows: list[dict[str, object]] = []
    for index, (selection_key, relative, payload) in enumerate(keyed, 1):
        path = source / "images" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "schema_version": 1,
                "index": index,
                "selection_key": selection_key,
                "source_relative_path": relative,
                "archive_relative_path": f"images/{relative}",
                "size_bytes": len(payload),
                "sha256": _sha(payload),
                "source_last_write_utc": timestamp,
            }
        )
    manifest = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\r\n").encode() for row in rows
    )
    (source / "manifest.jsonl").write_bytes(manifest)
    image_bytes = sum(int(row["size_bytes"]) for row in rows)
    contract = {
        "schema_version": 1,
        "kind": "otherimages_white_sync_dataset_v1",
        "source_semantics": "white_image_unlabeled_source_only",
        "source_root_at_capture": r"D:\download2\OtherImages",
        "source_files_modified": False,
        "selection": {
            "algorithm": "sha256_utf8_salt_lf_normalized_relative_path_sort_v1",
            "salt": salt,
            "normalized_path": "unicode_nfc_lowercase_forward_slashes",
            "supported_extensions": [".jpg", ".png", ".bmp", ".webp"],
            "candidate_count": 710_000,
            "requested_count": 4,
            "selected_count": 4,
            "prefix_stable_when_candidate_set_and_salt_are_unchanged": True,
        },
        "payload": {
            "images_root": "images",
            "manifest": "manifest.jsonl",
            "manifest_sha256": _sha(manifest),
            "image_count": 4,
            "image_bytes": image_bytes,
        },
        "labels": {
            "human_labels_present": False,
            "paddle_teacher_labels_present": False,
            "intended_next_step": "paddle_teacher_generation_then_frozen_split_training_and_validation",
        },
        "captured_utc": timestamp,
        "host": "windows-test-host",
        "powershell_version": "5.1",
    }
    if mutate_contract is not None:
        mutate_contract(contract)
    contract_raw = _json_bytes(contract)
    (source / "dataset.contract.json").write_bytes(contract_raw)
    subject = _sha(f"{_sha(manifest)}\n{_sha(contract_raw)}\n".encode())
    receipt = {
        "schema_version": 1,
        "kind": "otherimages_white_sync_receive_receipt_v1",
        "status": "complete",
        "download": {
            "url": "https://download.test/white-4.zip",
            "archive_path": str(tmp_path / "incoming" / "package.zip"),
            "size_bytes": 1234,
            "sha256": "a" * 64,
        },
        "package_subject_sha256": subject,
        "contract": {
            "path": "dataset.contract.json",
            "size_bytes": len(contract_raw),
            "sha256": _sha(contract_raw),
            "kind": "otherimages_white_sync_dataset_v1",
        },
        "manifest": {
            "path": "manifest.jsonl",
            "size_bytes": len(manifest),
            "sha256": _sha(manifest),
        },
        "verified_payload": {
            "image_count": 4,
            "image_bytes": image_bytes,
            "every_file_size_and_sha256_verified": True,
            "archive_file_closure_exact": True,
        },
        "publication": {
            "version": source.name,
            "raw_version_root": str(source.resolve()),
            "brand_new": True,
            "atomic_rename": True,
        },
        "completed_utc": timestamp,
    }
    receipt_path = tmp_path / "receive-evidence" / f"{source.name}.receive.receipt.json"
    receipt_path.parent.mkdir()
    receipt_path.write_bytes(_json_bytes(receipt))
    return source, receipt_path, rows, manifest, subject


def _materialize(tmp_path: Path, source: Path, receipt: Path, *, version: str = "pilot-v1"):
    return materializer.materialize_pilot(
        source_root=source,
        source_receipt=receipt,
        output_root=tmp_path / "pilot",
        evidence_root=tmp_path / "pilot-evidence",
        version=version,
    )


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fixed_production_shape_is_10k_to_first_1k() -> None:
    assert materializer.SOURCE_IMAGE_COUNT == 10_000
    assert materializer.PILOT_IMAGE_COUNT == 1_000
    parser = materializer._parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--source-count" not in option_strings
    assert "--prefix-count" not in option_strings


def test_materializes_exact_prefix_with_cross_bound_receipts(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, rows, full_manifest, source_subject = _build_publication(tmp_path)
    before = _snapshot(source)

    receipt = _materialize(tmp_path, source, source_receipt)

    assert _snapshot(source) == before
    output = tmp_path / "pilot" / "pilot-v1"
    prefix_manifest = b"".join(full_manifest.splitlines(keepends=True)[:2])
    assert (output / "manifest.jsonl").read_bytes() == prefix_manifest
    for row in rows[:2]:
        relative = Path(str(row["archive_relative_path"]))
        assert (output / relative).read_bytes() == (source / relative).read_bytes()
    for row in rows[2:]:
        assert not (output / Path(str(row["archive_relative_path"]))).exists()

    pilot_contract = json.loads((output / "dataset.contract.json").read_text())
    assert pilot_contract["analysis_only"] is True
    assert pilot_contract["production_route_authorized"] is False
    assert pilot_contract["payload"]["image_count"] == 2
    assert pilot_contract["payload"]["manifest_sha256"] == _sha(prefix_manifest)
    assert pilot_contract["source"]["package_subject_sha256"] == source_subject
    assert pilot_contract["source"]["full_manifest"]["sha256"] == _sha(full_manifest)

    internal = (output / "pilot.receipt.json").read_bytes()
    external = (tmp_path / "pilot-evidence" / "pilot-v1.pilot.receipt.json").read_bytes()
    assert internal == external
    assert receipt["source"]["package_subject_sha256"] == source_subject
    assert receipt["source"]["full_manifest"]["sha256"] == _sha(full_manifest)
    assert receipt["prefix"]["manifest"]["sha256"] == _sha(prefix_manifest)
    assert receipt["prefix"]["manifest"]["exact_byte_prefix_of_source_manifest"] is True
    assert receipt["validation"] == {
        "receive_receipt_contract_manifest_strict": True,
        "source_file_and_directory_closure_exact": True,
        "every_source_file_size_and_sha256_revalidated": True,
        "every_prefix_copy_size_and_sha256_verified": True,
        "source_files_written": False,
        "output_file_and_directory_closure_exact": True,
    }
    assert not list((tmp_path / "pilot").glob(".*.stage-*"))
    assert not list((tmp_path / "pilot-evidence").glob(".*.stage-*"))


def test_accepts_dotnet_round_trip_utc_without_changing_sealed_spelling(
    tmp_path: Path, small_counts: None
) -> None:
    timestamp = "2026-08-13T07:08:22.1879420Z"
    source, source_receipt, rows, full_manifest, _subject = _build_publication(
        tmp_path, timestamp=timestamp
    )
    source_before = _snapshot(source)

    _materialize(tmp_path, source, source_receipt)

    assert _snapshot(source) == source_before
    assert materializer._require_utc(timestamp, "fixture") == timestamp
    output = tmp_path / "pilot" / "pilot-v1"
    prefix_manifest = b"".join(full_manifest.splitlines(keepends=True)[:2])
    assert (output / "manifest.jsonl").read_bytes() == prefix_manifest
    assert all(row["source_last_write_utc"] == timestamp for row in rows)
    source_contract = json.loads((source / "dataset.contract.json").read_text())
    receive_receipt = json.loads(source_receipt.read_text())
    assert source_contract["captured_utc"] == timestamp
    assert receive_receipt["completed_utc"] == timestamp


def test_rehashes_nonprefix_files_and_never_publishes_tampered_source(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, rows, _manifest, _subject = _build_publication(tmp_path)
    nonprefix = source / Path(str(rows[-1]["archive_relative_path"]))
    original = nonprefix.read_bytes()
    nonprefix.write_bytes(b"x" * len(original))

    with pytest.raises(materializer.MaterializeError, match="size/SHA256 or stability mismatch"):
        _materialize(tmp_path, source, source_receipt)

    assert not (tmp_path / "pilot" / "pilot-v1").exists()
    assert not (tmp_path / "pilot-evidence" / "pilot-v1.pilot.receipt.json").exists()
    assert not list((tmp_path / "pilot").glob(".*.stage-*"))


def test_rejects_contract_count_drift_even_when_receipt_is_rebound(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, _rows, _manifest, _subject = _build_publication(
        tmp_path,
        mutate_contract=lambda contract: contract["selection"].__setitem__("requested_count", 3),
    )

    with pytest.raises(materializer.MaterializeError, match="exactly 4 selected images"):
        _materialize(tmp_path, source, source_receipt)

    assert not (tmp_path / "pilot" / "pilot-v1").exists()


def test_rejects_receive_receipt_schema_or_subject_drift(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, _rows, _manifest, _subject = _build_publication(tmp_path)
    document = json.loads(source_receipt.read_text())
    document["unreviewed_extra"] = True
    source_receipt.write_bytes(_json_bytes(document))
    with pytest.raises(materializer.MaterializeError, match="receive receipt keys differ"):
        _materialize(tmp_path, source, source_receipt)

    second = tmp_path / "second"
    source, source_receipt, _rows, _manifest, _subject = _build_publication(second)
    document = json.loads(source_receipt.read_text())
    document["package_subject_sha256"] = "0" * 64
    source_receipt.write_bytes(_json_bytes(document))
    with pytest.raises(materializer.MaterializeError, match="package subject mismatch"):
        _materialize(second, source, source_receipt)


def test_rejects_manifest_that_is_not_in_deterministic_selection_order(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, _rows, _manifest, _subject = _build_publication(
        tmp_path, reorder_manifest=True
    )

    with pytest.raises(materializer.MaterializeError, match="deterministic selection order"):
        _materialize(tmp_path, source, source_receipt)


def test_rejects_source_tree_extra_and_symlink_entries(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, _rows, _manifest, _subject = _build_publication(tmp_path)
    (source / "unexpected.txt").write_text("unexpected")
    with pytest.raises(materializer.MaterializeError, match="tree closure mismatch"):
        _materialize(tmp_path, source, source_receipt)

    second = tmp_path / "second"
    source, source_receipt, rows, _manifest, _subject = _build_publication(second)
    target = source / Path(str(rows[0]["archive_relative_path"]))
    linked = source / "images" / "linked.png"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(materializer.MaterializeError, match="symlink"):
        _materialize(second, source, source_receipt)


def test_refuses_to_overwrite_existing_output_or_receipt(
    tmp_path: Path, small_counts: None
) -> None:
    source, source_receipt, _rows, _manifest, _subject = _build_publication(tmp_path)
    existing = tmp_path / "pilot" / "pilot-v1"
    existing.mkdir(parents=True)
    marker = existing / "owner.txt"
    marker.write_text("keep")

    with pytest.raises(materializer.MaterializeError, match="already exists"):
        _materialize(tmp_path, source, source_receipt)
    assert marker.read_text() == "keep"

    second = tmp_path / "second"
    source, source_receipt, _rows, _manifest, _subject = _build_publication(second)
    evidence = second / "pilot-evidence"
    evidence.mkdir()
    receipt = evidence / "pilot-v1.pilot.receipt.json"
    receipt.write_text("owner")
    with pytest.raises(materializer.MaterializeError, match="receipt already exists"):
        _materialize(second, source, source_receipt)
    assert receipt.read_text() == "owner"
    assert not (second / "pilot" / "pilot-v1").exists()
