from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from transfer_receipt_ai.paddle_ocr_bundle import (
    CONTRACT_FILENAME,
    DELIVERY_CONTRACT_FILENAME,
    PaddleOcrBundleError,
    _paddle_onnx_options,
    _paddle_native_options,
    _require_dynamic_ocr_shapes,
    build_parser,
    package_delivery_bundle,
    snapshot_bundle,
    verify_bundle,
    verify_delivery_bundle,
)


def _model_dir(root: Path, role: str) -> Path:
    directory = root / role
    directory.mkdir(parents=True)
    (directory / "inference.pdmodel").write_bytes(f"{role}-model".encode())
    (directory / "inference.pdiparams").write_bytes(f"{role}-parameters".encode())
    (directory / "inference.yml").write_text(f"name: {role}\n", encoding="utf-8")
    return directory


def _snapshot(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    charset = source / "ppocr_keys_v1.txt"
    charset.parent.mkdir(parents=True)
    charset.write_text("你\n好\n", encoding="utf-8")
    return snapshot_bundle(
        output_dir=tmp_path / "bundle",
        model_dirs={role: _model_dir(source, role) for role in ("det", "rec", "cls")},
        charset_path=charset,
        effective_args={"lang": "ch", "use_angle_cls": True, "rec_image_shape": "3,48,320"},
        runtime={"paddleocr_version": "2.10.0", "resolved_by": "test"},
    )


def _add_fake_onnx_records(bundle: Path) -> None:
    contract_path = bundle / CONTRACT_FILENAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    records = {}
    for role in ("det", "rec", "cls"):
        path = bundle / "onnx" / f"paddle_ocr_{role}.onnx"
        path.parent.mkdir(exist_ok=True)
        payload = f"fake-{role}-onnx".encode()
        path.write_bytes(payload)
        records[role] = {
            "path": path.relative_to(bundle).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "io": {"inputs": [], "outputs": []},
            "dynamic_shape_validation": {},
        }
    contract["onnx"] = records
    contract_path.write_text(json.dumps(contract), encoding="utf-8")


def test_snapshot_bundle_copies_all_assets_and_records_adapter_contract(tmp_path: Path) -> None:
    bundle = _snapshot(tmp_path)

    contract = verify_bundle(bundle)

    assert (bundle / "paddle" / "det" / "inference.pdmodel").is_file()
    assert (bundle / "paddle" / "rec" / "inference.pdiparams").is_file()
    assert (bundle / "charset" / "ppocr_keys_v1.txt").read_text(encoding="utf-8") == "你\n好\n"
    assert contract["runtime"]["paddleocr_version"] == "2.10.0"
    assert contract["effective_paddleocr_args"]["use_angle_cls"] is True
    assert contract["adapter_contract"]["input_color_order"] == "RGB_passthrough_to_paddle_v2"
    assert contract["adapter_contract"]["adapter_version"] == "paddle_ocr_dotnet_adapter_v1"
    assert contract["adapter_contract"]["preprocessing"]["classifier_recognizer_right_padding"] == (
        "float_zero_after_normalization"
    )
    assert contract["native_asset_identity"]["kind"] == "paddle_ocr_native_asset_identity_v1"
    assert set(contract["native_asset_identity"]["components"]) == {"det", "rec", "cls", "dictionary"}
    assert contract["onnx"] == {}
    parsed = json.loads((bundle / CONTRACT_FILENAME).read_text(encoding="utf-8"))
    assert parsed["assets"]["cls"]["bundle_directory"] == "paddle/cls"


def test_verify_bundle_detects_modified_dictionary(tmp_path: Path) -> None:
    bundle = _snapshot(tmp_path)
    (bundle / "charset" / "ppocr_keys_v1.txt").write_text("被改了\n", encoding="utf-8")

    with pytest.raises(PaddleOcrBundleError, match="(size|SHA-256) differs"):
        verify_bundle(bundle)


def test_verify_bundle_detects_tampered_native_identity(tmp_path: Path) -> None:
    bundle = _snapshot(tmp_path)
    contract_path = bundle / CONTRACT_FILENAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["native_asset_identity"]["components"]["rec"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(PaddleOcrBundleError, match="native asset identity differs"):
        verify_bundle(bundle)


def test_snapshot_rejects_output_inside_source_model_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    charset = source / "ppocr_keys_v1.txt"
    charset.parent.mkdir(parents=True)
    charset.write_text("你\n", encoding="utf-8")
    model_dirs = {role: _model_dir(source, role) for role in ("det", "rec", "cls")}

    with pytest.raises(PaddleOcrBundleError, match="must not be inside"):
        snapshot_bundle(
            output_dir=model_dirs["det"] / "bundle",
            model_dirs=model_dirs,
            charset_path=charset,
            effective_args={},
            runtime={},
        )


def test_verify_can_require_all_three_onnx_files(tmp_path: Path) -> None:
    bundle = _snapshot(tmp_path)

    with pytest.raises(PaddleOcrBundleError, match="has not exported"):
        verify_bundle(bundle, require_onnx=True)


def test_package_delivery_copies_only_deployable_assets(tmp_path: Path) -> None:
    bundle = _snapshot(tmp_path)
    _add_fake_onnx_records(bundle)

    delivery = package_delivery_bundle(bundle_dir=bundle, output_dir=tmp_path / "delivery")
    contract = verify_delivery_bundle(delivery)

    assert not (delivery / "paddle").exists()
    assert (delivery / "onnx" / "paddle_ocr_rec.onnx").is_file()
    assert (delivery / "charset" / "ppocr_keys_v1.txt").is_file()
    assert (delivery / DELIVERY_CONTRACT_FILENAME).is_file()
    assert contract["effective_paddleocr_args"]["rec_model_dir"] == "onnx/paddle_ocr_rec.onnx"
    assert contract["package_size_bytes"] > 0
    assert contract["native_asset_identity"]["kind"] == "paddle_ocr_native_asset_identity_v1"


def test_verify_delivery_rejects_adapter_contract_semantic_change(tmp_path: Path) -> None:
    bundle = _snapshot(tmp_path)
    _add_fake_onnx_records(bundle)
    delivery = package_delivery_bundle(bundle_dir=bundle, output_dir=tmp_path / "delivery")
    contract_path = delivery / DELIVERY_CONTRACT_FILENAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["adapter_contract"]["preprocessing"]["classifier_recognizer_right_padding"] = "uint8_black"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(PaddleOcrBundleError, match="unsupported Paddle OCR adapter contract"):
        verify_delivery_bundle(delivery)


def test_export_rejects_lost_dynamic_detector_or_recognizer_axes() -> None:
    dynamic = {"inputs": [{"shape": [None, 3, "height", "width"]}]}
    assert _require_dynamic_ocr_shapes("det", dynamic)["required_dynamic_axes"] == [2, 3]
    assert _require_dynamic_ocr_shapes("rec", dynamic)["required_dynamic_axes"] == [3]

    with pytest.raises(PaddleOcrBundleError, match="lost required dynamic"):
        _require_dynamic_ocr_shapes("det", {"inputs": [{"shape": [1, 3, 960, 960]}]})


def test_bundle_cli_exposes_conversion_and_delivery_stages() -> None:
    parser = build_parser()
    args = parser.parse_args(["validate-onnx", "--bundle", "bundle", "--input", "images", "--output", "report"])

    assert args.min_text_exact_match == 1.0
    assert args.max_confidence_delta == 0.01


def test_conversion_parity_reader_is_forced_to_cpu_even_if_snapshot_used_gpu(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    contract = {
        "effective_paddleocr_args": {"lang": "ch", "use_gpu": True, "gpu_id": 7},
        "onnx": {
            role: {"path": f"onnx/paddle_ocr_{role}.onnx"}
            for role in ("det", "rec", "cls")
        },
        "dictionary": {"path": "charset/ppocr_keys_v1.txt"},
    }

    options = _paddle_onnx_options(bundle, contract)

    assert options["use_onnx"] is True
    assert options["use_gpu"] is False
    assert options["onnx_providers"] == ["CPUExecutionProvider"]
    assert options["det_model_dir"] == str(bundle / "onnx" / "paddle_ocr_det.onnx")


def test_conversion_native_reader_is_bound_to_snapshot_not_mutable_cache(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    contract = {
        "effective_paddleocr_args": {"lang": "ch", "use_gpu": True, "gpu_id": 7},
        "assets": {
            role: {"bundle_directory": f"paddle/{role}"}
            for role in ("det", "rec", "cls")
        },
        "onnx": {
            role: {"path": f"onnx/paddle_ocr_{role}.onnx"}
            for role in ("det", "rec", "cls")
        },
        "dictionary": {"path": "charset/ppocr_keys_v1.txt"},
    }

    options = _paddle_native_options(bundle, contract)

    assert options["use_onnx"] is False
    assert options["use_gpu"] is False
    assert "onnx_providers" not in options
    assert options["det_model_dir"] == str(bundle / "paddle" / "det")
    assert options["rec_model_dir"] == str(bundle / "paddle" / "rec")
    assert options["cls_model_dir"] == str(bundle / "paddle" / "cls")
    assert options["rec_char_dict_path"] == str(bundle / "charset" / "ppocr_keys_v1.txt")
