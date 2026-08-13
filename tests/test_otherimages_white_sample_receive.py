from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import stat
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "otherimages-white-sample-receive.py"
SPEC = importlib.util.spec_from_file_location("otherimages_white_sample_receive", SCRIPT)
assert SPEC and SPEC.loader
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package(
    tmp_path: Path,
    *,
    image_name: str = "one.png",
    image: bytes = b"image-data",
    archive_image: bytes | None = None,
    zip_separator: str = "/",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    row = {
        "schema_version": 1,
        "index": 1,
        "selection_key": _sha(b"selection"),
        "source_relative_path": image_name,
        "archive_relative_path": f"images/{image_name}",
        "size_bytes": len(image),
        "sha256": _sha(image),
        "source_last_write_utc": timestamp,
    }
    manifest = (json.dumps(row, separators=(",", ":")) + "\r\n").encode()
    contract = {
        "schema_version": 1,
        "kind": "otherimages_white_sync_dataset_v1",
        "source_semantics": "white_image_unlabeled_source_only",
        "source_root_at_capture": r"D:\download2\OtherImages",
        "source_files_modified": False,
        "selection": {
            "algorithm": "sha256_utf8_salt_lf_normalized_relative_path_sort_v1",
            "salt": "test",
            "normalized_path": "unicode_nfc_lowercase_forward_slashes",
            "supported_extensions": [".png"],
            "candidate_count": 1,
            "requested_count": 1,
            "selected_count": 1,
            "prefix_stable_when_candidate_set_and_salt_are_unchanged": True,
        },
        "payload": {
            "images_root": "images",
            "manifest": "manifest.jsonl",
            "manifest_sha256": _sha(manifest),
            "image_count": 1,
            "image_bytes": len(image),
        },
        "labels": {
            "human_labels_present": False,
            "paddle_teacher_labels_present": False,
            "intended_next_step": "paddle_teacher_generation_then_frozen_split_training_and_validation",
        },
        "captured_utc": timestamp,
        "host": "test-host",
        "powershell_version": "5.1",
    }
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.jsonl", manifest)
        output.writestr("dataset.contract.json", json.dumps(contract).encode())
        zip_image_name = f"images/{image_name}".replace("/", zip_separator)
        output.writestr(zip_image_name, image if archive_image is None else archive_image)
    return archive


def _receive(tmp_path: Path, archive: Path, **kwargs):
    data = archive.read_bytes()
    expected_sha = kwargs.pop("expected_sha", _sha(data))

    def sealed_download(url, destination, expected_archive_sha256, expected_archive_bytes, timeout_seconds):
        assert url == "https://download.test/package.zip"
        if len(data) != expected_archive_bytes:
            raise receiver.ReceiveError(
                f"download size mismatch: expected {expected_archive_bytes}, got {len(data)}"
            )
        actual_sha = _sha(data)
        if actual_sha != expected_archive_sha256:
            raise receiver.ReceiveError(
                f"download SHA256 mismatch: expected {expected_archive_sha256}, got {actual_sha}"
            )
        destination.write_bytes(data)

    with patch.object(receiver, "_download", side_effect=sealed_download):
        return receiver.receive_package(
            url="https://download.test/package.zip",
            expected_archive_sha256=expected_sha,
            expected_archive_bytes=len(data),
            incoming_root=tmp_path / "incoming",
            raw_root=tmp_path / "raw",
            evidence_root=tmp_path / "evidence",
            version=kwargs.pop("version", "sample-v1"),
            **kwargs,
        )


def test_receives_verifies_and_atomically_publishes(tmp_path: Path):
    archive = _package(tmp_path)
    receipt = _receive(tmp_path, archive)

    published = tmp_path / "raw" / "sample-v1"
    assert (published / "images" / "one.png").read_bytes() == b"image-data"
    assert (published / "manifest.jsonl").is_file()
    assert receipt["verified_payload"] == {
        "image_count": 1,
        "image_bytes": 10,
        "every_file_size_and_sha256_verified": True,
        "archive_file_closure_exact": True,
    }
    persisted = json.loads((tmp_path / "evidence" / "sample-v1.receive.receipt.json").read_text())
    assert persisted["package_subject_sha256"] == receipt["package_subject_sha256"]
    assert not list((tmp_path / "raw").glob(".*.stage-*"))


def test_receives_official_windows_backslash_zip_entries(tmp_path: Path):
    archive = _package(tmp_path, image_name="nested/one.png", zip_separator="\\")
    receipt = _receive(tmp_path, archive)

    published = tmp_path / "raw" / "sample-v1"
    assert (published / "images" / "nested" / "one.png").read_bytes() == b"image-data"
    assert receipt["verified_payload"]["archive_file_closure_exact"] is True


def test_accepts_powershell_round_trip_utc_timestamp():
    timestamp = "2026-08-11T14:45:06.7724197Z"
    assert receiver._require_utc_timestamp(timestamp, "timestamp") == timestamp


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape",
        "/absolute",
        "//server/share/escape",
        r"\absolute",
        r"\\server\share\escape",
        r"C:\escape",
        r"C:escape",
        "images/../escape",
        r"images\..\escape",
        "images//escape",
        r"images\\escape",
    ],
)
def test_rejects_unsafe_zip_paths_without_publication(tmp_path: Path, bad_name: str):
    archive = _package(tmp_path)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(bad_name, b"bad")
    with pytest.raises(
        receiver.ReceiveError,
        match=r"(?:unsafe|absolute/UNC|drive-qualified) path",
    ):
        _receive(tmp_path, archive)
    assert not (tmp_path / "raw" / "sample-v1").exists()


@pytest.mark.parametrize("bad_name", [r"images/mixed\escape", r"images\mixed/escape"])
def test_rejects_mixed_zip_path_separators(tmp_path: Path, bad_name: str):
    archive = _package(tmp_path)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(bad_name, b"bad")
    with pytest.raises(receiver.ReceiveError, match="mixed ZIP path separators"):
        _receive(tmp_path, archive)


def test_rejects_zip_separator_normalization_collision(tmp_path: Path):
    archive = _package(tmp_path)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(r"images\one.png", b"duplicate-normalized-name")
    with pytest.raises(receiver.ReceiveError, match="collision"):
        _receive(tmp_path, archive)


@pytest.mark.parametrize(
    "directory_kind",
    ["forward_trailing", "backslash_trailing", "unix_mode", "dos_attribute"],
)
def test_rejects_every_zip_directory_representation(
    tmp_path: Path, directory_kind: str
):
    archive = _package(tmp_path)
    with zipfile.ZipFile(archive, "a") as output:
        if directory_kind == "forward_trailing":
            output.writestr("extra/", b"")
        elif directory_kind == "backslash_trailing":
            output.writestr("extra\\", b"")
        elif directory_kind == "unix_mode":
            directory = zipfile.ZipInfo("extra")
            directory.create_system = 3
            directory.external_attr = (stat.S_IFDIR | 0o755) << 16
            output.writestr(directory, b"")
        else:
            directory = zipfile.ZipInfo("extra")
            directory.create_system = 0
            directory.external_attr = 0x10
            output.writestr(directory, b"")
    with pytest.raises(receiver.ReceiveError, match="directory entry is forbidden"):
        _receive(tmp_path, archive)
    assert not (tmp_path / "raw" / "sample-v1").exists()


def test_rejects_symlink_and_case_collision(tmp_path: Path):
    archive = _package(tmp_path)
    with zipfile.ZipFile(archive, "a") as output:
        link = zipfile.ZipInfo("images/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, "one.png")
    with pytest.raises(receiver.ReceiveError, match="symlink"):
        _receive(tmp_path, archive)

    archive = _package(tmp_path / "second", image_name="One.png")
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("images/one.png", b"other")
    with pytest.raises(receiver.ReceiveError, match="collision"):
        _receive(tmp_path / "second", archive)


def test_rejects_payload_hash_and_never_clobbers_existing_version(tmp_path: Path):
    archive = _package(tmp_path, archive_image=b"tampered")
    with pytest.raises(receiver.ReceiveError):
        _receive(tmp_path, archive)
    assert not (tmp_path / "raw" / "sample-v1").exists()

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_archive = _package(clean_root)
    existing = clean_root / "raw" / "sample-v1"
    existing.mkdir(parents=True)
    marker = existing / "marker"
    marker.write_text("keep")
    with pytest.raises(receiver.ReceiveError, match="already exists"):
        _receive(clean_root, clean_archive)
    assert marker.read_text() == "keep"


def test_rejects_transport_binding_before_extracting(tmp_path: Path):
    archive = _package(tmp_path)
    with pytest.raises(receiver.ReceiveError, match="SHA256 mismatch"):
        _receive(tmp_path, archive, expected_sha="0" * 64)
    assert not any((tmp_path / "raw").iterdir())


def test_rejects_prefixed_or_mixed_layout(tmp_path: Path):
    archive = _package(tmp_path)
    rewritten = tmp_path / "prefixed.zip"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(rewritten, "w") as output:
        for info in source.infolist():
            output.writestr(f"package/{info.filename}", source.read(info))
    with pytest.raises(receiver.ReceiveError, match="root manifest.jsonl"):
        _receive(tmp_path, rewritten)

    archive = _package(tmp_path / "mixed")
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("package/manifest.jsonl", b"not-accepted")
    with pytest.raises(receiver.ReceiveError, match="closure differs"):
        _receive(tmp_path / "mixed", archive)


def test_enforces_archive_and_uncompressed_limits_before_publication(tmp_path: Path):
    archive = _package(tmp_path)
    archive_bytes = archive.stat().st_size
    with pytest.raises(receiver.ReceiveError, match="archive exceeds limit"):
        _receive(tmp_path, archive, max_archive_bytes=archive_bytes - 1)
    with pytest.raises(receiver.ReceiveError, match="uncompressed payload exceeds limit"):
        _receive(tmp_path / "expanded", archive, max_uncompressed_bytes=1)
