from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai import otherimages_paddle_capture as capture_module
from transfer_receipt_ai import otherimages_paddle_teacher as teacher_module
from transfer_receipt_ai.otherimages_inventory import build_otherimages_inventory
from transfer_receipt_ai.otherimages_paddle_capture import (
    PaddleCaptureBatch,
    PaddleCapturedLine,
    PaddleViewContract,
    capture_paddle_view,
    capture_paddle_three_views,
)
from transfer_receipt_ai.otherimages_paddle_teacher import (
    ADAPTER_EVIDENCE_KIND,
    PINNED_ADAPTER_IMPLEMENTATION,
    TeacherContractError,
    build_paddle_teacher_consensus,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _inventory(tmp_path: Path, *, image_count: int = 1) -> Path:
    source = tmp_path / "OtherImages"
    source.mkdir()
    for index in range(image_count):
        pixels = np.full((300 + index * 17, 180 + index * 11, 3), 255, dtype=np.uint8)
        pixels[20:30, 10 : 80 + index] = 0
        pixels[100:106, 20 : 140 + index * 3] = 30
        Image.fromarray(pixels).save(source / f"receipt-{index}.png")
    inventory = tmp_path / "inventory"
    build_otherimages_inventory(input_dir=source, output_dir=inventory, layout_sample_size=4)
    return inventory


def _asset_binding(path: Path) -> dict[str, object]:
    if path.is_file():
        root = path.parent
        files = [path]
    else:
        root = path
        files = sorted(item for item in path.rglob("*") if item.is_file())
    bindings = [
        {
            "path": item.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            "size_bytes": item.stat().st_size,
        }
        for item in files
    ]
    return {
        "path": str(path.resolve()),
        "files": bindings,
        "closure_sha256": teacher_module._canonical_sha256(bindings),
        "size_bytes": sum(int(item["size_bytes"]) for item in bindings),
    }


def _adapter_evidence(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "paddle-model-fixture"
    root.mkdir(exist_ok=True)
    assets: dict[str, dict[str, object]] = {}
    for role in ("det", "cls", "rec"):
        role_root = root / role
        role_root.mkdir(exist_ok=True)
        model_file = role_root / "inference.pdiparams"
        if not model_file.exists():
            model_file.write_bytes(f"fixture-{role}".encode("ascii"))
        assets[role] = _asset_binding(role_root)
    dictionary = root / "chars.txt"
    if not dictionary.exists():
        dictionary.write_text("甲\n乙\n", encoding="utf-8", newline="\n")
    assets["dictionary"] = _asset_binding(dictionary)
    runtime_versions = {
        "paddleocr": "2.10.0",
        "paddlepaddle": "3.0.0-fixture",
        "albumentations": "1.4.10",
        "albucore": "0.0.13",
        "opencv": "4.10.0-fixture",
        "numpy": "1.26.4-fixture",
        "pillow": "10.3.0-fixture",
    }
    effective_args = {
        "ocr_version": "PP-OCRv4",
        "det_algorithm": "DB",
        "det_limit_side_len": 960,
        "det_limit_type": "max",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        "det_db_score_mode": "fast",
        "det_box_type": "quad",
        "rec_algorithm": "SVTR_LCNet",
        "rec_image_shape": "3, 48, 320",
        "rec_batch_num": 6,
        "max_text_length": 25,
        "use_space_char": True,
        "cls_image_shape": "3, 48, 192",
        "cls_batch_num": 6,
        "cls_thresh": 0.9,
        "use_angle_cls": True,
        "drop_score": 0.5,
        "use_onnx": False,
        "precision": "fp32",
        "use_tensorrt": False,
        "enable_mkldnn": False,
        "cpu_threads": 10,
        "use_gpu": False,
        "gpu_id": 0,
    }
    model_payload = {
        "adapter_implementation": PINNED_ADAPTER_IMPLEMENTATION,
        "paddleocr_version": "2.10.0",
        "runtime_versions": runtime_versions,
        "effective_paddle_args": effective_args,
        "device": "cpu",
        "drop_score": 0.5,
        "assets": assets,
        "adapter_input_color_bridge": "opencv_rgb8_to_bgr8_v1",
    }
    return {
        "kind": ADAPTER_EVIDENCE_KIND,
        "adapter_implementation": PINNED_ADAPTER_IMPLEMENTATION,
        "paddle_version": "2.10.0",
        "model_contract_sha256": teacher_module._canonical_sha256(model_payload),
        "drop_score": 0.5,
        "stages": {"db": True, "cls": True, "rec": True},
        "execution_device": "cpu",
        "runtime_versions": runtime_versions,
        "effective_paddle_args": effective_args,
        "model_assets": assets,
        "raw_db_lines_preserved_before_drop_filter": True,
        "adapter_input_color_bridge": "opencv_rgb8_to_bgr8_v1",
    }


class _FakeAdapter:
    def __init__(self, tmp_path: Path) -> None:
        self._evidence = _adapter_evidence(tmp_path)

    def evidence(self) -> dict[str, object]:
        return self._evidence

    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        assert transformed_rgb.dtype == np.uint8
        assert not transformed_rgb.flags.writeable
        assert view.operations
        line = PaddleCapturedLine(
                text="ＡＢＣ １２３",
                confidence=0.99,
                quad_normalized=((0.1, 0.1), (0.8, 0.1), (0.8, 0.2), (0.1, 0.2)),
            )
        return PaddleCaptureBatch((line,), 1, 1, 0)


def test_injected_adapter_captures_three_views_consumable_by_offline_consensus(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    capture_directory = tmp_path / "captured-views"
    capture_directory.mkdir()
    captured: list[Path] = []
    for view_id in ("original_rgb", "grayscale_clahe", "upscale_sharpen"):
        output = capture_directory / f"{view_id}.jsonl"
        receipt = capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id=view_id,
            operations=None,
            adapter=_FakeAdapter(tmp_path),
        )
        assert receipt["records"] == 1
        assert receipt["capture_errors"] == 0
        assert receipt["paddle_imported_by_repository_core"] is False
        row = _read_jsonl(output)[0]
        assert row["capture_state"] == "ok"
        assert row["adapter"]["stages"] == {"db": True, "cls": True, "rec": True}
        assert row["lines"][0]["passes_drop_score"] is True
        assert row["view_contract"]["quad_coordinate_space"] == "exif_upright_source_normalized"
        captured.append(output)

    teacher_output = tmp_path / "teacher"
    contract = build_paddle_teacher_consensus(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        view_results=captured,
        output_dir=teacher_output,
    )

    assert contract["counts"]["accepted_teacher_records"] == 1
    teacher = _read_jsonl(teacher_output / "teacher_manifest.jsonl")[0]
    assert teacher["text"] == "ABC 123"
    assert teacher["consensus"]["agreement"] == "3_of_3"


def test_batch_capture_uses_one_adapter_instance_and_publishes_all_three_views(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "three-view-capture"
    adapter = _FakeAdapter(tmp_path)

    receipt = capture_paddle_three_views(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        output_dir=output,
        adapter=adapter,
    )

    assert receipt["kind"] == "otherimages_paddle_three_view_capture_receipt_v1"
    assert receipt["records_per_view"] == 1
    assert receipt["capture_errors"] == 0
    assert {path.name for path in output.iterdir()} == {
        "original_rgb.jsonl",
        "grayscale_clahe.jsonl",
        "upscale_sharpen.jsonl",
    }
    model_contracts = {
        _read_jsonl(path)[0]["adapter"]["model_contract_sha256"]
        for path in output.iterdir()
    }
    assert len(model_contracts) == 1

    teacher_output = tmp_path / "teacher-from-batch"
    contract = build_paddle_teacher_consensus(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        view_results=sorted(output.iterdir()),
        output_dir=teacher_output,
    )
    assert contract["counts"]["accepted_teacher_records"] == 1


class _SelectiveFailureAdapter(_FakeAdapter):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.calls = 0

    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("fixture GPU decode failure")
        return super().capture(transformed_rgb, view)


def test_adapter_failure_is_a_capture_error_row_and_other_records_continue(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, image_count=2)
    output = tmp_path / "view.jsonl"

    receipt = capture_paddle_view(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        output_path=output,
        view_id="original_rgb",
        operations=None,
        adapter=_SelectiveFailureAdapter(tmp_path),
    )
    rows = _read_jsonl(output)

    assert receipt["records"] == 2
    assert receipt["capture_errors"] == 1
    assert sorted(row["capture_state"] for row in rows) == ["error", "ok"]
    failed = next(row for row in rows if row["capture_state"] == "error")
    assert failed["lines"] is None
    assert "fixture GPU decode failure" in failed["error"]


class _InvalidEvidenceAdapter(_FakeAdapter):
    def evidence(self) -> dict[str, object]:
        value = super().evidence()
        value["stages"] = {"db": True, "cls": False, "rec": True}
        return value


def test_adapter_without_db_cls_rec_evidence_is_fatal_before_capture(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "must-not-publish.jsonl"

    with pytest.raises(TeacherContractError, match=r"DB\+CLS\+REC"):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=_InvalidEvidenceAdapter(tmp_path),
        )

    assert not output.exists()


class _MutatingAdapter(_FakeAdapter):
    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        transformed_rgb[0, 0] = 0
        return super().capture(transformed_rgb, view)


def test_adapter_cannot_mutate_core_transformed_pixels_and_error_is_captured(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "must-not-publish.jsonl"

    receipt = capture_paddle_view(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        output_path=output,
        view_id="original_rgb",
        operations=None,
        adapter=_MutatingAdapter(tmp_path),
    )
    assert receipt["capture_errors"] == 1
    assert _read_jsonl(output)[0]["capture_state"] == "error"


class _ReverseOrderAdapter(_FakeAdapter):
    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        del transformed_rgb, view
        bottom = PaddleCapturedLine(
            text="底部",
            confidence=0.99,
            quad_normalized=((0.1, 0.7), (0.8, 0.7), (0.8, 0.8), (0.1, 0.8)),
        )
        top = PaddleCapturedLine(
            text="顶部",
            confidence=0.99,
            quad_normalized=((0.1, 0.1), (0.8, 0.1), (0.8, 0.2), (0.1, 0.2)),
        )
        return PaddleCaptureBatch((bottom, top), 2, 2, 0)


def test_capture_core_canonicalises_line_order_independently_of_adapter(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "ordered.jsonl"

    capture_paddle_view(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        output_path=output,
        view_id="original_rgb",
        operations=None,
        adapter=_ReverseOrderAdapter(tmp_path),
    )
    lines = _read_jsonl(output)[0]["lines"]

    assert [line["text"] for line in lines] == ["顶部", "底部"]
    assert [line["index"] for line in lines] == [0, 1]
    assert [line["adapter_index"] for line in lines] == [1, 0]


class _AssetTamperingAdapter(_FakeAdapter):
    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        assets = self._evidence["model_assets"]
        det_root = Path(str(assets["det"]["path"]))
        (det_root / "inference.pdiparams").write_bytes(b"tampered-during-capture")
        return super().capture(transformed_rgb, view)


def test_model_asset_changed_during_capture_prevents_publication(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "must-not-publish.jsonl"

    with pytest.raises(TeacherContractError, match="model asset"):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=_AssetTamperingAdapter(tmp_path),
        )

    assert not output.exists()


def test_capture_output_cannot_be_nested_in_inventory_source_root(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "OtherImages" / "captured.jsonl"

    with pytest.raises(TeacherContractError, match="source_root"):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=_FakeAdapter(tmp_path),
        )

    assert not output.exists()


def test_batch_output_cannot_stage_inside_inventory_source_root(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    source_root = tmp_path / "OtherImages"
    before = sorted(path.name for path in source_root.iterdir())
    output = source_root / "three-view-capture"

    with pytest.raises(TeacherContractError, match="source_root"):
        capture_paddle_three_views(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_dir=output,
            adapter=_FakeAdapter(tmp_path),
        )

    assert not output.exists()
    assert sorted(path.name for path in source_root.iterdir()) == before
    assert not list(source_root.glob(".three-view-capture.capture-building-*"))


def test_capture_atomic_no_replace_race_preserves_competitor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "raced.jsonl"
    original_publish = capture_module._rename_directory_no_replace

    def race(
        stage: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_stage_identity: tuple[int, int] | None = None,
    ) -> None:
        destination.write_text("competitor\n", encoding="utf-8")
        original_publish(
            stage,
            destination,
            expected_parent_identity=expected_parent_identity,
            expected_stage_identity=expected_stage_identity,
        )

    monkeypatch.setattr(capture_module, "_rename_directory_no_replace", race)
    with pytest.raises(FileExistsError):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=_FakeAdapter(tmp_path),
        )

    assert output.read_text(encoding="utf-8") == "competitor\n"
    failed_stages = list(tmp_path.glob(".raced.jsonl.capture-building-*"))
    assert len(failed_stages) == 1
    assert failed_stages[0].is_file()


def test_capture_rejects_output_parent_replacement_during_ocr(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output_parent = tmp_path / "capture-publication-parent"
    output_parent.mkdir()
    moved_parent = tmp_path / "capture-publication-parent-before-swap"
    output = output_parent / "view.jsonl"

    class ParentReplacingAdapter(_FakeAdapter):
        def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
            output_parent.rename(moved_parent)
            output_parent.mkdir()
            return super().capture(transformed_rgb, view)

    with pytest.raises(TeacherContractError, match="output parent identity changed"):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=ParentReplacingAdapter(tmp_path),
        )

    assert not output.exists()
    assert not list(output_parent.glob(".view.jsonl.capture-building-*"))


def test_capture_publish_parent_swap_preserves_stage_and_same_named_unrelated_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(tmp_path)
    output_parent = tmp_path / "capture-publish-parent"
    output_parent.mkdir()
    moved_parent = tmp_path / "capture-publish-parent-before-swap"
    output = output_parent / "view.jsonl"
    original_publish = capture_module._rename_directory_no_replace
    stage_name: str | None = None

    def swap_parent_then_publish(
        stage: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_stage_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal stage_name
        stage_name = stage.name
        output_parent.rename(moved_parent)
        output_parent.mkdir()
        (output_parent / stage.name).write_text("unrelated; never delete", encoding="utf-8")
        original_publish(
            stage,
            destination,
            expected_parent_identity=expected_parent_identity,
            expected_stage_identity=expected_stage_identity,
        )

    monkeypatch.setattr(capture_module, "_rename_directory_no_replace", swap_parent_then_publish)
    with pytest.raises(RuntimeError, match="output parent identity changed"):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=_FakeAdapter(tmp_path),
        )

    assert stage_name is not None
    assert (moved_parent / stage_name).is_file()
    assert (output_parent / stage_name).read_text(encoding="utf-8") == "unrelated; never delete"
    assert not output.exists()


def test_batch_publish_parent_swap_never_recursively_deletes_unrelated_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(tmp_path)
    output_parent = tmp_path / "batch-publish-parent"
    output_parent.mkdir()
    moved_parent = tmp_path / "batch-publish-parent-before-swap"
    output = output_parent / "three-views"
    original_publish = capture_module._rename_directory_no_replace
    stage_name: str | None = None

    def swap_final_parent_then_publish(
        stage: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_stage_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal stage_name
        if destination != output:
            original_publish(
                stage,
                destination,
                expected_parent_identity=expected_parent_identity,
                expected_stage_identity=expected_stage_identity,
            )
            return
        stage_name = stage.name
        output_parent.rename(moved_parent)
        output_parent.mkdir()
        imposter = output_parent / stage.name
        imposter.mkdir()
        (imposter / "unrelated.txt").write_text("preserve me", encoding="utf-8")
        original_publish(
            stage,
            destination,
            expected_parent_identity=expected_parent_identity,
            expected_stage_identity=expected_stage_identity,
        )

    monkeypatch.setattr(capture_module, "_rename_directory_no_replace", swap_final_parent_then_publish)
    with pytest.raises(RuntimeError, match="output parent identity changed"):
        capture_paddle_three_views(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_dir=output,
            adapter=_FakeAdapter(tmp_path),
        )

    assert stage_name is not None
    assert (moved_parent / stage_name / "original_rgb.jsonl").is_file()
    assert (output_parent / stage_name / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"
    assert not output.exists()


def test_capture_refuses_same_parent_stage_file_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "captured.jsonl"
    original_publish = capture_module._rename_directory_no_replace
    original_stage: Path | None = None
    replacement_stage: Path | None = None

    def replace_stage_then_publish(
        stage: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_stage_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal original_stage, replacement_stage
        original_stage = stage.with_name(stage.name + "-bound-original")
        stage.rename(original_stage)
        stage.write_text("attacker replacement", encoding="utf-8")
        replacement_stage = stage
        original_publish(
            stage,
            destination,
            expected_parent_identity=expected_parent_identity,
            expected_stage_identity=expected_stage_identity,
        )

    monkeypatch.setattr(capture_module, "_rename_directory_no_replace", replace_stage_then_publish)
    with pytest.raises(RuntimeError, match="stage identity changed"):
        capture_paddle_view(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            output_path=output,
            view_id="original_rgb",
            operations=None,
            adapter=_FakeAdapter(tmp_path),
        )

    assert original_stage is not None and original_stage.is_file()
    assert replacement_stage is not None
    assert replacement_stage.read_text(encoding="utf-8") == "attacker replacement"
    assert not output.exists()


def test_capture_checkout_wrapper_documents_injected_factory_and_view_contract() -> None:
    wrapper = Path(__file__).parents[1] / "scripts" / "otherimages-paddle-capture.py"
    result = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--adapter-factory ADAPTER_FACTORY" in result.stdout
    assert "--operation OPERATION" in result.stdout
    assert "{original_rgb,grayscale_clahe,upscale_sharpen,all}" in result.stdout
    assert "all (default)" in result.stdout
    assert "PaddleOCR" in result.stdout
    assert "2.10.0 DB+CLS+REC adapter" in result.stdout
