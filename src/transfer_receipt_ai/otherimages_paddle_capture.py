"""Adapter boundary for capturing one Paddle DB+CLS+REC OtherImages view.

The repository includes a pinned Windows PaddleOCR 2.10.0 factory and also
allows a contract-compatible factory through ``--adapter-factory``.  Paddle is
imported only when that factory is instantiated, never by the offline teacher
aggregator.

The adapter must return line quadrilaterals in EXIF-upright source-normalized
coordinates even if its view transform resizes, crops, pads, or enhances the
pixels.  This makes geometry comparable across the three sealed view files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .otherimages_inventory import (
    _bind_stage_identity,
    _paths_overlap,
    _rename_directory_no_replace,
    _require_no_reparse_ancestors,
)
from .otherimages_paddle_teacher import (
    CANONICAL_VIEW_OPERATIONS,
    CAPTURE_KIND,
    SCHEMA_VERSION,
    TeacherContractError,
    _canonical_sha256,
    _bind_output_parent,
    canonical_view_contract,
    _observe_adapter_assets,
    _public_binding,
    _read_bound_file,
    _verify_adapter_assets,
    _verify_output_parent,
    _verify_observation,
    _validate_adapter_evidence,
    load_inventory_for_teacher,
)


CAPTURE_RECEIPT_KIND = "otherimages_paddle_layout_capture_receipt_v2"
THREE_VIEW_CAPTURE_RECEIPT_KIND = "otherimages_paddle_three_view_capture_receipt_v2"


@dataclass(frozen=True)
class PaddleCapturedLine:
    """One DB box with its CLS-corrected REC text and confidence."""

    text: str
    confidence: float
    orientation_degrees: int
    transformed_quad_pixels: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    quad_normalized: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


@dataclass(frozen=True)
class PaddleCaptureBatch:
    """Raw DB coverage plus one CLS+REC result for every detected line."""

    lines: tuple[PaddleCapturedLine, ...]
    raw_detected_line_count: int
    recognition_attempted_line_count: int
    recognition_rejected_line_count: int


@dataclass(frozen=True)
class PaddleViewContract:
    """Immutable view operations understood by the injected Windows adapter."""

    view_id: str
    operations: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        payload = canonical_view_contract(self.view_id)
        if tuple(payload["operations"]) != self.operations:
            raise TeacherContractError(f"operations do not match canonical {self.view_id} recipe")
        return payload


@runtime_checkable
class PaddleLayoutCaptureAdapter(Protocol):
    """Implemented by a separate Windows/Paddle worker package."""

    def evidence(self) -> Mapping[str, object]:
        """Return version/model/drop-score evidence with DB, CLS, and REC true."""

    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        """Run DB+CLS+REC on immutable core-transformed RGB pixels."""


@dataclass(frozen=True)
class PreparedPaddleCapture:
    """Validated inventory/model/view state shared by one or three captures."""

    inventory_rows: tuple[dict[str, object], ...]
    inventory_observations: tuple[dict[str, object], ...]
    adapter_evidence: dict[str, object]
    model_asset_observations: tuple[dict[str, object], ...]
    output_paths: tuple[Path, ...]
    output_parent_identities: tuple[tuple[int, int], ...]
    source_roots: tuple[Path, ...]


@dataclass(frozen=True)
class PreparedSourceImage:
    """Immutable upright pixels and byte observation loaded once per source."""

    rgb: np.ndarray
    observation: dict[str, object]


def _load_adapter_factory(specification: str) -> PaddleLayoutCaptureAdapter:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise TeacherContractError("adapter factory must use MODULE:CALLABLE syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TeacherContractError(f"adapter factory is not callable: {specification}")
    adapter = factory()
    if not isinstance(adapter, PaddleLayoutCaptureAdapter):
        raise TeacherContractError(
            "adapter factory result must expose callable evidence() and capture(source_image, view)"
        )
    return adapter


def _finite_confidence(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("line confidence must be numeric")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("line confidence must be finite and in [0,1]")
    return confidence


def _pixel_sha256(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _load_bound_upright_rgb(
    path: Path,
    *,
    expected_raw_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    data, observation = _read_bound_file(path, description="inventory source image")
    if observation["sha256"] != expected_raw_sha256:
        raise TeacherContractError(
            f"source image raw SHA-256 differs from inventory: {observation['path']}; "
            f"expected={expected_raw_sha256}, observed={observation['sha256']}"
        )
    with io.BytesIO(data) as stream:
        with Image.open(stream) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb), observation


def _canonical_transform(source_rgb: np.ndarray, view: PaddleViewContract) -> np.ndarray:
    if view.view_id == "original_rgb":
        transformed = source_rgb.copy()
    elif view.view_id == "grayscale_clahe":
        grayscale = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
        transformed = np.repeat(enhanced[:, :, None], 3, axis=2)
    elif view.view_id == "upscale_sharpen":
        height, width = source_rgb.shape[:2]
        resized = cv2.resize(source_rgb, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (5, 5), sigmaX=1.0, sigmaY=1.0)
        transformed = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0.0)
    else:  # pragma: no cover - PaddleViewContract payload already rejects this.
        raise TeacherContractError(f"unsupported canonical view: {view.view_id}")
    return np.ascontiguousarray(transformed, dtype=np.uint8)


def _capture_lines(
    adapter: PaddleLayoutCaptureAdapter,
    *,
    transformed_rgb: np.ndarray,
    view: PaddleViewContract,
    drop_score: float,
) -> tuple[list[dict[str, object]], int, int, int]:
    immutable_rgb = np.ascontiguousarray(transformed_rgb, dtype=np.uint8)
    immutable_rgb.setflags(write=False)
    batch = adapter.capture(immutable_rgb, view)
    if not isinstance(batch, PaddleCaptureBatch):
        raise TypeError("adapter capture result must be PaddleCaptureBatch")
    for name, value in (
        ("raw_detected_line_count", batch.raw_detected_line_count),
        ("recognition_attempted_line_count", batch.recognition_attempted_line_count),
        ("recognition_rejected_line_count", batch.recognition_rejected_line_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"adapter {name} must be a non-negative integer")
    raw_lines = batch.lines
    if batch.raw_detected_line_count != len(raw_lines):
        raise ValueError("adapter must return one line record for every raw DB box")
    if batch.recognition_attempted_line_count != batch.raw_detected_line_count:
        raise ValueError("adapter must attempt recognition for every raw DB box")
    if batch.recognition_rejected_line_count > batch.recognition_attempted_line_count:
        raise ValueError("adapter rejected-line count cannot exceed attempted-line count")
    rows: list[dict[str, object]] = []
    for index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, PaddleCapturedLine):
            raise TypeError(f"adapter line {index} must be PaddleCapturedLine")
        if not isinstance(raw_line.text, str):
            raise TypeError(f"adapter line {index} text must be a string")
        confidence = _finite_confidence(raw_line.confidence)
        orientation_degrees = raw_line.orientation_degrees
        if (
            isinstance(orientation_degrees, bool)
            or not isinstance(orientation_degrees, int)
            or orientation_degrees not in {0, 180}
        ):
            raise ValueError(f"adapter line {index} orientation_degrees must be 0 or 180")
        if len(raw_line.transformed_quad_pixels) != 4:
            raise ValueError(f"adapter line {index} transformed_quad_pixels must contain four points")
        transformed_quad_pixels: list[list[float]] = []
        for point_index, point in enumerate(raw_line.transformed_quad_pixels):
            if len(point) != 2:
                raise ValueError(f"adapter line {index} transformed point {point_index} must be [x,y]")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or x < 0.0 or y < 0.0:
                raise ValueError(f"adapter line {index} transformed point {point_index} is invalid")
            transformed_quad_pixels.append([x, y])
        if len(raw_line.quad_normalized) != 4:
            raise ValueError(f"adapter line {index} quad must contain four points")
        quad: list[list[float]] = []
        for point_index, point in enumerate(raw_line.quad_normalized):
            if len(point) != 2:
                raise ValueError(f"adapter line {index} point {point_index} must be [x,y]")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(f"adapter line {index} point {point_index} is outside source-normalized [0,1]")
            quad.append([x, y])
        transformed_height, transformed_width = immutable_rgb.shape[:2]
        for point_index, (transformed_point, normalized_point) in enumerate(
            zip(transformed_quad_pixels, quad)
        ):
            transformed_x, transformed_y = transformed_point
            if transformed_x > transformed_width - 1 or transformed_y > transformed_height - 1:
                raise ValueError(f"adapter line {index} transformed point {point_index} is outside chosen view")
            expected_x = transformed_x / max(1, transformed_width - 1)
            expected_y = transformed_y / max(1, transformed_height - 1)
            if not (
                math.isclose(normalized_point[0], expected_x, rel_tol=0.0, abs_tol=1e-7)
                and math.isclose(normalized_point[1], expected_y, rel_tol=0.0, abs_tol=1e-7)
            ):
                raise ValueError(
                    f"adapter line {index} point {point_index} normalized geometry differs from transformed pixels"
                )
        rows.append(
            {
                "index": index,
                "text": raw_line.text,
                "confidence": confidence,
                "passes_drop_score": confidence >= drop_score,
                "orientation_degrees": orientation_degrees,
                "transformed_quad_pixels": transformed_quad_pixels,
                "quad_normalized": quad,
            }
        )
    rows.sort(
        key=lambda row: (
            sum(float(point[1]) for point in row["quad_normalized"]) / 4.0,
            sum(float(point[0]) for point in row["quad_normalized"]) / 4.0,
            int(row["index"]),
        )
    )
    for canonical_index, row in enumerate(rows):
        row["adapter_index"] = row["index"]
        row["index"] = canonical_index
    return (
        rows,
        batch.raw_detected_line_count,
        batch.recognition_attempted_line_count,
        batch.recognition_rejected_line_count,
    )


def _write_capture_file(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _prepare_capture(
    *,
    inventory_manifest: Path,
    output_paths: Sequence[Path],
    adapter: PaddleLayoutCaptureAdapter,
    inventory_contract: Path | None,
) -> PreparedPaddleCapture:
    if not isinstance(adapter, PaddleLayoutCaptureAdapter):
        raise TeacherContractError("adapter must implement evidence() and capture()")
    adapter_evidence = _validate_adapter_evidence(adapter.evidence(), location="capture adapter")
    model_asset_observations = tuple(_observe_adapter_assets(adapter_evidence, location="capture preflight"))
    inventory_rows, _inventory_payload, inventory_observations_list = load_inventory_for_teacher(
        inventory_manifest,
        contract_path=inventory_contract,
    )
    pending = [row for row in inventory_rows if row["teacher_state"] == "pending"]
    if not pending:
        raise TeacherContractError("inventory contains no pending Paddle capture rows")
    inventory_observations = tuple(inventory_observations_list)
    inventory_directory = Path(str(inventory_observations[0]["path"])).parent
    source_roots = tuple(
        sorted(
            {
                Path(str(row["source_root"])).expanduser().resolve(strict=True)
                for row in inventory_rows
            },
            key=os.fspath,
        )
    )
    normalized_outputs: list[Path] = []
    output_parent_identities: list[tuple[int, int]] = []
    for raw_output in output_paths:
        output = Path(os.path.abspath(os.path.expanduser(os.fspath(raw_output))))
        _require_no_reparse_ancestors(output, include_leaf=False)
        if output.exists():
            raise FileExistsError(f"capture output must be brand-new: {output}")
        if not output.parent.is_dir():
            raise NotADirectoryError(f"capture output parent must already exist: {output.parent}")
        if any(_paths_overlap(output, Path(str(item["path"]))) for item in inventory_observations):
            raise TeacherContractError("capture output must be disjoint from inventory evidence")
        if _paths_overlap(output, inventory_directory):
            raise TeacherContractError("capture output must be disjoint from the complete inventory publication directory")
        if any(_paths_overlap(output, source_root) for source_root in source_roots):
            raise TeacherContractError("capture output must be disjoint from every inventory source_root")
        model_assets = adapter_evidence["model_assets"]
        assert isinstance(model_assets, Mapping)
        for role, asset_value in model_assets.items():
            assert isinstance(asset_value, Mapping)
            asset_path = Path(str(asset_value["path"])).expanduser().resolve(strict=True)
            protected_path = asset_path if asset_path.is_dir() else asset_path.parent
            if _paths_overlap(output, protected_path):
                raise TeacherContractError(f"capture output must be disjoint from Paddle {role} model assets")
        normalized_outputs.append(output)
        output_parent_identities.append(_bind_output_parent(output))
    if len({os.path.normcase(os.fspath(path)) for path in normalized_outputs}) != len(normalized_outputs):
        raise TeacherContractError("capture outputs must be distinct")
    for left_index, left in enumerate(normalized_outputs):
        for right in normalized_outputs[left_index + 1 :]:
            if _paths_overlap(left, right):
                raise TeacherContractError("capture outputs must be mutually disjoint")
    return PreparedPaddleCapture(
        inventory_rows=tuple(inventory_rows),
        inventory_observations=inventory_observations,
        adapter_evidence=adapter_evidence,
        model_asset_observations=model_asset_observations,
        output_paths=tuple(normalized_outputs),
        output_parent_identities=tuple(output_parent_identities),
        source_roots=source_roots,
    )


def capture_paddle_view(
    *,
    inventory_manifest: Path,
    output_path: Path,
    view_id: str,
    operations: Sequence[str] | None,
    adapter: PaddleLayoutCaptureAdapter,
    inventory_contract: Path | None = None,
    _prepared: PreparedPaddleCapture | None = None,
    _prepared_sources: Mapping[str, PreparedSourceImage] | None = None,
) -> dict[str, object]:
    """Capture one complete view file suitable for the offline aggregator."""
    if not view_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in view_id):
        raise TeacherContractError("view_id must use lowercase ASCII letters/digits/._- and start with a letter")
    if not view_id[0].isalpha() or not view_id[0].isascii() or len(view_id) > 64:
        raise TeacherContractError("view_id must use lowercase ASCII letters/digits/._- and start with a letter")
    canonical_payload = canonical_view_contract(view_id)
    canonical_operations = tuple(str(operation) for operation in canonical_payload["operations"])
    normalized_operations = canonical_operations if operations is None else tuple(operations)
    if normalized_operations != canonical_operations:
        raise TeacherContractError(f"operations must exactly match the canonical {view_id} recipe")
    prepared = _prepared or _prepare_capture(
        inventory_manifest=inventory_manifest,
        output_paths=[output_path],
        adapter=adapter,
        inventory_contract=inventory_contract,
    )
    if len(prepared.output_paths) == 1:
        requested_output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
        output = prepared.output_paths[0] if _prepared is None else requested_output
    else:
        requested_output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
        if requested_output not in prepared.output_paths:
            raise TeacherContractError("capture output is outside the prepared batch")
        output = requested_output
    _require_no_reparse_ancestors(output, include_leaf=False)
    if output.exists():
        raise FileExistsError(f"capture output must be brand-new: {output}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"capture output parent must already exist: {output.parent}")
    output_parent_identity = (
        prepared.output_parent_identities[0]
        if _prepared is None
        else _bind_output_parent(output)
    )
    adapter_evidence = prepared.adapter_evidence
    model_asset_observations = list(prepared.model_asset_observations)
    view = PaddleViewContract(view_id=view_id, operations=normalized_operations)
    view_payload = view.payload()
    view_sha256 = _canonical_sha256(view_payload)

    inventory_rows = list(prepared.inventory_rows)
    inventory_observations = list(prepared.inventory_observations)
    pending = [row for row in inventory_rows if row["teacher_state"] == "pending"]
    if not pending:
        raise TeacherContractError("inventory contains no pending Paddle capture rows")
    source_observations: list[dict[str, object]] = []
    captured_rows: list[dict[str, object]] = []
    drop_score = float(adapter_evidence["drop_score"])
    for inventory in sorted(pending, key=lambda item: str(item["record_id"])):
        source = Path(str(inventory["source_absolute_path"]))
        if _paths_overlap(output, source.parent):
            raise TeacherContractError("capture output must be disjoint from source image directories")
        prepared_source = (
            _prepared_sources.get(str(inventory["record_id"]))
            if _prepared_sources is not None
            else None
        )
        if prepared_source is None:
            source_rgb, source_observation = _load_bound_upright_rgb(
                source,
                expected_raw_sha256=str(inventory["raw_sha256"]),
            )
        else:
            source_rgb = prepared_source.rgb
            source_observation = prepared_source.observation
            _verify_observation(source_observation, description="prepared source image before Paddle capture")
        source_observations.append(source_observation)
        source_decoded_sha256 = _pixel_sha256(source_rgb)
        if source_decoded_sha256 != inventory["decoded_pixel_sha256"]:
            raise TeacherContractError(f"source decoded pixels differ from inventory before capture: {source}")
        transformed_rgb = _canonical_transform(source_rgb, view)
        transformed_sha256 = _pixel_sha256(transformed_rgb)
        try:
            lines, raw_detected_count, attempted_count, rejected_count = _capture_lines(
                adapter,
                transformed_rgb=transformed_rgb,
                view=view,
                drop_score=drop_score,
            )
            capture_state = "ok"
            error_payload: dict[str, object] = {}
        except Exception as error:
            lines = None
            raw_detected_count = None
            attempted_count = None
            rejected_count = None
            capture_state = "error"
            error_payload = {
                "error": f"{type(error).__name__}: {str(error)[:1024]}",
            }
        _verify_observation(source_observation, description="source image after Paddle capture")
        captured_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": CAPTURE_KIND,
                "inventory_manifest_sha256": inventory_observations[0]["sha256"],
                "inventory_contract_sha256": inventory_observations[1]["sha256"],
                "view_id": view_id,
                "view_contract": view_payload,
                "view_contract_sha256": view_sha256,
                "adapter": adapter_evidence,
                "record_id": inventory["record_id"],
                "group_id": inventory["group_id"],
                "raw_sha256": inventory["raw_sha256"],
                "decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
                "transform_receipt": {
                    "view_id": view_id,
                    "view_contract_sha256": view_sha256,
                    "source_decoded_pixel_sha256": source_decoded_sha256,
                    "transformed_pixel_sha256": transformed_sha256,
                    "source_width": int(source_rgb.shape[1]),
                    "source_height": int(source_rgb.shape[0]),
                    "transformed_width": int(transformed_rgb.shape[1]),
                    "transformed_height": int(transformed_rgb.shape[0]),
                    "coordinate_mapping": "full_frame_scale_source_normalized_identity_v1",
                },
                "capture_state": capture_state,
                "lines": lines,
                "raw_detected_line_count": raw_detected_count,
                "recognition_attempted_line_count": attempted_count,
                "recognition_rejected_line_count": rejected_count,
                **error_payload,
            }
        )

    _verify_adapter_assets(
        adapter_evidence,
        model_asset_observations,
        location="capture final input closure",
    )
    if _validate_adapter_evidence(adapter.evidence(), location="capture adapter after execution") != adapter_evidence:
        raise TeacherContractError("capture adapter evidence changed during execution")
    for observation in inventory_observations + source_observations + model_asset_observations:
        _verify_observation(observation, description="capture input at final closure")
    _verify_output_parent(output, output_parent_identity, location="capture staging")
    descriptor, stage_name = tempfile.mkstemp(prefix=f".{output.name}.capture-building-", dir=output.parent)
    os.close(descriptor)
    stage = Path(stage_name)
    stage.unlink()
    stage_identity: tuple[int, int] | None = None
    published = False
    try:
        _write_capture_file(stage, captured_rows)
        stage_identity = _bind_stage_identity(stage, directory=False)
        staged_data = stage.read_bytes()
        staged_binding = {
            "sha256": hashlib.sha256(staged_data).hexdigest(),
            "size_bytes": len(staged_data),
            "line_count": staged_data.count(b"\n"),
        }
        _verify_adapter_assets(
            adapter_evidence,
            model_asset_observations,
            location="capture publication",
        )
        if _validate_adapter_evidence(adapter.evidence(), location="capture adapter before publication") != adapter_evidence:
            raise TeacherContractError("capture adapter evidence changed before publication")
        for observation in inventory_observations + source_observations + model_asset_observations:
            _verify_observation(observation, description="capture input before publication")
        _verify_output_parent(output, output_parent_identity, location="capture publication")
        _rename_directory_no_replace(
            stage,
            output,
            expected_parent_identity=output_parent_identity,
            expected_stage_identity=stage_identity,
        )
        if stage_identity is None or _bind_stage_identity(output, directory=False) != stage_identity:
            raise TeacherContractError("published Paddle capture differs from bound stage")
        published = True
    finally:
        if not published:
            # Failure stages are retained as evidence; pathname cleanup is
            # unsafe after a parent/stage replacement race.
            pass
    data = output.read_bytes()
    published_binding = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }
    if published_binding != staged_binding:
        raise TeacherContractError("published Paddle capture failed exact readback")
    _verify_output_parent(output, output_parent_identity, location="capture post-publication readback")
    if stage_identity is None or _bind_stage_identity(output, directory=False) != stage_identity:
        raise TeacherContractError("published Paddle capture identity changed during readback")
    if _validate_adapter_evidence(adapter.evidence(), location="capture adapter after publication") != adapter_evidence:
        raise TeacherContractError("capture adapter evidence changed after publication")
    _verify_adapter_assets(
        adapter_evidence,
        model_asset_observations,
        location="capture post-publication readback",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CAPTURE_RECEIPT_KIND,
        "view_id": view_id,
        "view_contract_sha256": view_sha256,
        "adapter": adapter_evidence,
        "inventory_manifest": _public_binding(inventory_observations[0]),
        "inventory_contract": _public_binding(inventory_observations[1]),
        "records": len(captured_rows),
        "capture_errors": sum(row["capture_state"] == "error" for row in captured_rows),
        "output": {
            "path": str(output),
            **published_binding,
        },
        "paddle_imported_by_repository_core": False,
    }


def capture_paddle_three_views(
    *,
    inventory_manifest: Path,
    output_dir: Path,
    adapter: PaddleLayoutCaptureAdapter,
    inventory_contract: Path | None = None,
) -> dict[str, object]:
    """Capture all three canonical views from one frozen adapter/model instance."""
    output_directory = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    _require_no_reparse_ancestors(output_directory, include_leaf=False)
    if output_directory.exists():
        raise FileExistsError(f"three-view output directory must be brand-new: {output_directory}")
    if not output_directory.parent.is_dir():
        raise NotADirectoryError(f"three-view output parent must already exist: {output_directory.parent}")
    output_parent_identity = _bind_output_parent(output_directory)
    prepared = _prepare_capture(
        inventory_manifest=inventory_manifest,
        output_paths=[output_directory],
        adapter=adapter,
        inventory_contract=inventory_contract,
    )
    _verify_output_parent(output_directory, output_parent_identity, location="three-view capture staging")
    stage = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.capture-building-", dir=output_directory.parent))
    stage_identity = _bind_stage_identity(stage, directory=True)
    published = False
    try:
        view_ids = tuple(CANONICAL_VIEW_OPERATIONS)
        stage_outputs = tuple(stage / f"{view_id}.jsonl" for view_id in view_ids)
        prepared_sources: dict[str, PreparedSourceImage] = {}
        for inventory in sorted(
            (row for row in prepared.inventory_rows if row["teacher_state"] == "pending"),
            key=lambda row: str(row["record_id"]),
        ):
            source_rgb, observation = _load_bound_upright_rgb(
                Path(str(inventory["source_absolute_path"])),
                expected_raw_sha256=str(inventory["raw_sha256"]),
            )
            source_rgb.setflags(write=False)
            prepared_sources[str(inventory["record_id"])] = PreparedSourceImage(
                rgb=source_rgb,
                observation=observation,
            )
        receipts = [
            capture_paddle_view(
                inventory_manifest=inventory_manifest,
                inventory_contract=inventory_contract,
                output_path=stage_output,
                view_id=view_id,
                operations=None,
                adapter=adapter,
                _prepared=prepared,
                _prepared_sources=prepared_sources,
            )
            for view_id, stage_output in zip(view_ids, stage_outputs)
        ]
        if output_directory.exists():
            raise FileExistsError(f"three-view output directory appeared during capture: {output_directory}")
        _verify_output_parent(output_directory, output_parent_identity, location="three-view capture publication")
        _rename_directory_no_replace(
            stage,
            output_directory,
            expected_parent_identity=output_parent_identity,
            expected_stage_identity=stage_identity,
        )
        if _bind_stage_identity(output_directory, directory=True) != stage_identity:
            raise TeacherContractError("published three-view capture differs from bound stage")
        expected_members = {f"{view_id}.jsonl" for view_id in view_ids}
        if {path.name for path in output_directory.iterdir()} != expected_members:
            raise TeacherContractError("published three-view capture membership differs after publication")
        outputs = []
        for receipt, view_id in zip(receipts, view_ids):
            output = output_directory / f"{view_id}.jsonl"
            data = output.read_bytes()
            binding = {
                "path": str(output),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "line_count": data.count(b"\n"),
            }
            if binding["sha256"] != dict(receipt["output"])["sha256"]:
                raise TeacherContractError(f"three-view published readback differs for {view_id}")
            outputs.append({"view_id": view_id, **binding})
        _verify_output_parent(
            output_directory,
            output_parent_identity,
            location="three-view capture post-publication readback",
        )
        if _bind_stage_identity(output_directory, directory=True) != stage_identity:
            raise TeacherContractError("published three-view capture identity changed during readback")
        published = True
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": THREE_VIEW_CAPTURE_RECEIPT_KIND,
            "views": outputs,
            "adapter": prepared.adapter_evidence,
            "records_per_view": receipts[0]["records"],
            "capture_errors": sum(int(receipt["capture_errors"]) for receipt in receipts),
            "output_directory": str(output_directory),
        }
    finally:
        if not published:
            # Preserve the exact failed stage for forensic inspection.
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pinned Windows PaddleOCR 2.10.0 DB+CLS+REC view capture; "
            "the offline teacher aggregator never imports Paddle"
        )
    )
    parser.add_argument("--inventory", type=Path, required=True, help="inventory paddle_teacher_pending.jsonl")
    parser.add_argument("--inventory-contract", type=Path, help="sibling inventory.contract.json by default")
    parser.add_argument("--output", type=Path, required=True, help="brand-new captured view JSONL")
    parser.add_argument(
        "--view-id",
        choices=[*CANONICAL_VIEW_OPERATIONS, "all"],
        default="all",
        help="canonical view to capture, or all (default) for the sealed three-view directory",
    )
    parser.add_argument(
        "--operation",
        action="append",
        help="optional exact canonical recipe assertion; omit to use the built-in recipe",
    )
    parser.add_argument(
        "--adapter-factory",
        default="transfer_receipt_ai.otherimages_paddle_v2_adapter:create_adapter",
        help=(
            "Windows/Paddle adapter factory in MODULE:CALLABLE form; default is the repository's pinned "
            "PaddleOCR 2.10.0 DB+CLS+REC adapter"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the complete capture receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    try:
        adapter = _load_adapter_factory(arguments.adapter_factory)
        if arguments.view_id == "all":
            if arguments.operation:
                raise TeacherContractError("--operation is valid only for a single canonical --view-id")
            receipt = capture_paddle_three_views(
                inventory_manifest=arguments.inventory,
                inventory_contract=arguments.inventory_contract,
                output_dir=arguments.output,
                adapter=adapter,
            )
        else:
            receipt = capture_paddle_view(
                inventory_manifest=arguments.inventory,
                inventory_contract=arguments.inventory_contract,
                output_path=arguments.output,
                view_id=arguments.view_id,
                operations=arguments.operation,
                adapter=adapter,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OtherImages Paddle view capture failed:\n{error}") from None
    if arguments.json:
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    elif receipt["kind"] == THREE_VIEW_CAPTURE_RECEIPT_KIND:
        print(
            f"Captured three canonical Paddle views for {receipt['records_per_view']} record(s); "
            f"errors={receipt['capture_errors']}; output={receipt['output_directory']}"
        )
    else:
        print(
            f"Captured {receipt['records']} record(s) for {receipt['view_id']}; "
            f"errors={receipt['capture_errors']}; output={dict(receipt['output'])['path']}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
