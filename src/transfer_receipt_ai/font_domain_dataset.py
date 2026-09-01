"""Validated document/region manifests for font-domain consistency work.

One JSONL row represents one receipt image and owns all of its text-region
crops.  This is intentionally different from ordinary crop classification:
the source image, every crop-margin variant and every compression derivative
must share one ``source_group_id`` and therefore one split.
One group may contain multiple source-image hashes when they are repeated
captures of the same underlying receipt; the split and font domain must still
remain identical across the whole group.

The manifest labels a *rendering domain* (for example ``ios_alipay``), not an
exact font file.  ``unknown`` is forbidden as a training label; it is produced
only by inference-time rejection.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from PIL import Image, ImageOps

from .font_domain import SCHEMA_VERSION, UNKNOWN_DOMAIN


DOCUMENT_KIND: Final[str] = "receipt_font_domain_document_v1"
REGION_RECORD_KIND: Final[str] = "receipt_font_domain_region_record_v1"
DATASET_KIND: Final[str] = "receipt_font_domain_dataset_v1"
PERCEPTUAL_HASH_ABI: Final[str] = "dct-phash-64-v1"
ALLOWED_SPLITS: Final[tuple[str, ...]] = ("train", "calibration", "test", "inference")
_DOMAIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "id",
        "source_group_id",
        "content_group_id",
        "split",
        "font_domain",
        "label_source",
        "device_prior_domain",
        "source_image_sha256",
        "regions",
    }
)
_REGION_FIELDS = frozenset(
    {
        "id",
        "role",
        "image",
        "include_in_consistency",
        "text",
        "raw_sha256",
        "pixel_sha256",
    }
)
DEFAULT_MAXIMUM_MANIFEST_BYTES: Final[int] = 64 * 1024 * 1024
DEFAULT_MAXIMUM_LINE_BYTES: Final[int] = 4 * 1024 * 1024
DEFAULT_MAXIMUM_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
DEFAULT_MAXIMUM_IMAGE_PIXELS: Final[int] = 20_000_000
DEFAULT_MAXIMUM_DOCUMENTS: Final[int] = 100_000
DEFAULT_MAXIMUM_REGIONS_PER_DOCUMENT: Final[int] = 1_000


def _require_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value.strip()


def _require_identifier(value: object, *, description: str) -> str:
    identifier = _require_string(value, description=description)
    if value != identifier or not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(
            f"{description} must match {_IDENTIFIER_PATTERN.pattern!r}; found {identifier!r}"
        )
    return identifier


def _require_domain(value: object, *, description: str) -> str:
    domain = _require_string(value, description=description)
    if value != domain:
        raise ValueError(f"{description} must use a canonical unpadded domain name")
    if domain == UNKNOWN_DOMAIN:
        raise ValueError(f"{description} cannot be {UNKNOWN_DOMAIN!r}; UNKNOWN is inference-only")
    if not _DOMAIN_PATTERN.fullmatch(domain):
        raise ValueError(
            f"{description} must match {_DOMAIN_PATTERN.pattern!r}; found {domain!r}"
        )
    return domain


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _read_bounded_bytes(path: Path, *, maximum_bytes: int, description: str) -> bytes:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    try:
        declared_size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"unable to stat {description} {path}: {error}") from error
    if declared_size > maximum_bytes:
        raise ValueError(
            f"{description} exceeds the {maximum_bytes}-byte limit: {path} ({declared_size} bytes)"
        )
    with path.open("rb") as stream:
        data = stream.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError(f"{description} exceeds the {maximum_bytes}-byte limit: {path}")
    return data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pixel_sha256(rgb: np.ndarray) -> str:
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in pixels.shape)).encode("ascii"))
    digest.update(b"\0uint8\0RGB\0")
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _perceptual_hash(rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low >= median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _load_upright_rgb(
    path: Path,
    data: bytes,
    *,
    maximum_pixels: int = DEFAULT_MAXIMUM_IMAGE_PIXELS,
) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = (int(value) for value in opened.size)
            if width < 1 or height < 1 or width * height > maximum_pixels:
                raise ValueError(
                    f"decoded dimensions {width}x{height} exceed the {maximum_pixels}-pixel limit"
                )
            image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to decode region image {path}: {error}") from error
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError(f"decoded region image is empty or invalid: {path}")
    return np.ascontiguousarray(rgb)


def _safe_image_path(dataset_root: Path, relative: object, *, location: str) -> tuple[str, Path]:
    value = _require_string(relative, description=f"image at {location}")
    if (
        "\\" in value
        or value.startswith("/")
        or ":" in value.split("/", 1)[0]
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"image at {location} must be a safe POSIX relative path")
    path = (dataset_root / Path(*value.split("/"))).resolve(strict=True)
    try:
        path.relative_to(dataset_root)
    except ValueError:
        raise ValueError(f"image at {location} escapes the dataset root") from None
    if not path.is_file():
        raise FileNotFoundError(path)
    return value, path


@dataclass(frozen=True)
class FontDomainRegion:
    region_id: str
    role: str
    relative_image: str
    image_path: Path
    include_in_consistency: bool
    text: str | None
    raw_sha256: str
    pixel_sha256: str
    perceptual_hash: str
    width: int
    height: int

    def load_bound_rgb(self) -> np.ndarray:
        """Reload this crop only if it still matches the validated byte/pixel snapshot."""

        data = _read_bounded_bytes(
            self.image_path,
            maximum_bytes=DEFAULT_MAXIMUM_IMAGE_BYTES,
            description="region image",
        )
        if _sha256_bytes(data) != self.raw_sha256:
            raise ValueError(f"region image bytes changed after validation: {self.image_path}")
        rgb = _load_upright_rgb(self.image_path, data)
        if _pixel_sha256(rgb) != self.pixel_sha256:
            raise ValueError(f"region image pixels changed after validation: {self.image_path}")
        return rgb

    def classifier_record(self, document: "FontDomainDocument") -> dict[str, object]:
        if document.font_domain is None:
            raise ValueError("cannot export an unlabeled inference document for classifier training")
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": REGION_RECORD_KIND,
            "id": f"{document.document_id}:{self.region_id}",
            "image": self.relative_image,
            "field": "font_domain",
            "class_name": document.font_domain,
            "split": "val" if document.split == "calibration" else document.split,
            "group_id": document.source_group_id,
            "document_id": document.document_id,
            "content_group_id": document.content_group_id,
            "region_id": self.region_id,
            "role": self.role,
            "include_in_consistency": self.include_in_consistency,
            "raw_sha256": self.raw_sha256,
            "pixel_sha256": self.pixel_sha256,
            "perceptual_hash": self.perceptual_hash,
            "width": self.width,
            "height": self.height,
            "label_source": document.label_source,
        }


@dataclass(frozen=True)
class FontDomainDocument:
    document_id: str
    source_group_id: str
    content_group_id: str | None
    split: str
    font_domain: str | None
    label_source: str | None
    device_prior_domain: str | None
    source_image_sha256: str | None
    regions: tuple[FontDomainRegion, ...]

    @property
    def included_regions(self) -> tuple[FontDomainRegion, ...]:
        return tuple(region for region in self.regions if region.include_in_consistency)


@dataclass(frozen=True)
class FontDomainDataset:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    documents: tuple[FontDomainDocument, ...]

    @property
    def snapshot_sha256(self) -> str:
        """Bind parsed labels/groups to every validated crop byte and pixel hash."""

        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": DATASET_KIND,
            "manifest_sha256": self.manifest_sha256,
            "documents": [
                {
                    "id": document.document_id,
                    "source_group_id": document.source_group_id,
                    "content_group_id": document.content_group_id,
                    "split": document.split,
                    "font_domain": document.font_domain,
                    "label_source": document.label_source,
                    "device_prior_domain": document.device_prior_domain,
                    "source_image_sha256": document.source_image_sha256,
                    "regions": [
                        {
                            "id": region.region_id,
                            "role": region.role,
                            "image": region.relative_image,
                            "include_in_consistency": region.include_in_consistency,
                            "text": region.text,
                            "raw_sha256": region.raw_sha256,
                            "pixel_sha256": region.pixel_sha256,
                            "perceptual_hash": region.perceptual_hash,
                            "width": region.width,
                            "height": region.height,
                        }
                        for region in sorted(document.regions, key=lambda value: value.region_id)
                    ],
                }
                for document in self.documents
            ],
        }
        return _sha256_bytes(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )

    def summary(self) -> dict[str, object]:
        split_counts = Counter(document.split for document in self.documents)
        domain_counts = Counter(
            document.font_domain for document in self.documents if document.font_domain is not None
        )
        region_count = sum(len(document.regions) for document in self.documents)
        included_count = sum(len(document.included_regions) for document in self.documents)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": DATASET_KIND,
            "manifest": self.manifest_path.as_posix(),
            "manifest_sha256": self.manifest_sha256,
            "dataset_snapshot_sha256": self.snapshot_sha256,
            "documents": len(self.documents),
            "regions": region_count,
            "included_regions": included_count,
            "source_groups": len({document.source_group_id for document in self.documents}),
            "content_groups": len(
                {document.content_group_id for document in self.documents if document.content_group_id}
            ),
            "splits": dict(sorted(split_counts.items())),
            "font_domains": dict(sorted((str(key), value) for key, value in domain_counts.items())),
            "unknown_is_training_class": False,
            "perceptual_hash_abi": PERCEPTUAL_HASH_ABI,
            "authenticity": "not_assessed",
        }


def load_font_domain_dataset(
    records_path: Path,
    *,
    require_labels: bool | None = None,
    minimum_regions: int = 1,
    require_leakage_metadata: bool = False,
    maximum_manifest_bytes: int = DEFAULT_MAXIMUM_MANIFEST_BYTES,
    maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES,
    maximum_image_bytes: int = DEFAULT_MAXIMUM_IMAGE_BYTES,
    maximum_image_pixels: int = DEFAULT_MAXIMUM_IMAGE_PIXELS,
    maximum_documents: int = DEFAULT_MAXIMUM_DOCUMENTS,
    maximum_regions_per_document: int = DEFAULT_MAXIMUM_REGIONS_PER_DOCUMENT,
) -> FontDomainDataset:
    """Load and bind one document-level JSONL manifest.

    Structural or binding failures are fail-closed before any output is
    created.  Exact decoded-pixel duplicates are forbidden across splits.
    """

    if isinstance(minimum_regions, bool) or not isinstance(minimum_regions, int) or minimum_regions < 1:
        raise ValueError("minimum_regions must be positive")
    if require_labels is not None and not isinstance(require_labels, bool):
        raise ValueError("require_labels must be true, false, or None")
    if not isinstance(require_leakage_metadata, bool):
        raise ValueError("require_leakage_metadata must be boolean")
    for name, limit in (
        ("maximum_manifest_bytes", maximum_manifest_bytes),
        ("maximum_line_bytes", maximum_line_bytes),
        ("maximum_image_bytes", maximum_image_bytes),
        ("maximum_image_pixels", maximum_image_pixels),
        ("maximum_documents", maximum_documents),
        ("maximum_regions_per_document", maximum_regions_per_document),
    ):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    manifest_path = records_path.expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    dataset_root = manifest_path.parent.resolve(strict=True)
    documents: list[FontDomainDocument] = []
    document_ids: set[str] = set()
    image_paths: set[str] = set()
    group_bindings: dict[str, tuple[str, str | None]] = {}
    content_splits: dict[str, str] = {}
    source_hash_bindings: dict[str, tuple[str, str, str | None]] = {}
    pixel_bindings: dict[str, tuple[str, str, str | None, str]] = {}

    manifest_bytes = _read_bounded_bytes(
        manifest_path,
        maximum_bytes=maximum_manifest_bytes,
        description="font-domain manifest",
    )
    try:
        manifest_text = manifest_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{manifest_path}: manifest must be UTF-8: {error}") from None

    for line_number, line in enumerate(manifest_text.splitlines(), start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > maximum_line_bytes:
                raise ValueError(
                    f"{manifest_path}:{line_number}: record exceeds the {maximum_line_bytes}-byte limit"
                )
            if len(documents) >= maximum_documents:
                raise ValueError(f"manifest exceeds the {maximum_documents}-document limit")
            try:
                raw: Any = json.loads(
                    line,
                    object_pairs_hook=_strict_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(raw, Mapping):
                raise ValueError(f"{manifest_path}:{line_number}: record must be an object")
            location = f"{manifest_path}:{line_number}"
            unknown_document_fields = sorted(set(raw) - _DOCUMENT_FIELDS)
            if unknown_document_fields:
                raise ValueError(
                    f"{location}: unknown document fields: {', '.join(unknown_document_fields)}"
                )
            raw_schema_version = raw.get("schema_version")
            if (
                isinstance(raw_schema_version, bool)
                or not isinstance(raw_schema_version, int)
                or raw_schema_version != SCHEMA_VERSION
                or raw.get("kind") != DOCUMENT_KIND
            ):
                raise ValueError(f"{location}: unsupported schema_version/kind")
            document_id = _require_identifier(raw.get("id"), description=f"id at {location}")
            if document_id in document_ids:
                raise ValueError(f"{location}: duplicate document id {document_id!r}")
            source_group_id = _require_identifier(
                raw.get("source_group_id"), description=f"source_group_id at {location}"
            )
            split = _require_string(raw.get("split"), description=f"split at {location}")
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"{location}: split must be one of {ALLOWED_SPLITS}")
            raw_domain = raw.get("font_domain")
            font_domain = None if raw_domain is None else _require_domain(
                raw_domain, description=f"font_domain at {location}"
            )
            if require_labels is True and font_domain is None:
                raise ValueError(f"{location}: font_domain is required")
            if require_labels is False and split != "inference":
                raise ValueError(f"{location}: unlabeled analysis records must use split='inference'")
            if split == "inference" and font_domain is not None:
                raise ValueError(f"{location}: inference records cannot contain font_domain")
            if split != "inference" and font_domain is None:
                raise ValueError(f"{location}: labeled train/calibration/test records require font_domain")
            if split == "inference" and require_labels is True:
                raise ValueError(f"{location}: inference records cannot be used to fit a model")
            label_source = raw.get("label_source")
            if font_domain is not None:
                label_source = _require_string(label_source, description=f"label_source at {location}")
            elif label_source is not None:
                label_source = _require_string(label_source, description=f"label_source at {location}")
            if split == "inference" and label_source is not None:
                raise ValueError(f"{location}: inference records cannot contain label_source")
            content_group_id = raw.get("content_group_id")
            if content_group_id is not None:
                content_group_id = _require_identifier(
                    content_group_id, description=f"content_group_id at {location}"
                )
                prior_content_split = content_splits.setdefault(content_group_id, split)
                if prior_content_split != split:
                    raise ValueError(
                        f"{location}: content_group_id {content_group_id!r} crosses "
                        f"{prior_content_split}/{split} splits"
                    )
            device_prior = raw.get("device_prior_domain")
            if device_prior is not None:
                device_prior = _require_domain(
                    device_prior, description=f"device_prior_domain at {location}"
                )
            source_image_sha256 = raw.get("source_image_sha256")
            if source_image_sha256 is not None:
                source_image_sha256 = _require_string(
                    source_image_sha256, description=f"source_image_sha256 at {location}"
                )
                if not _SHA256_PATTERN.fullmatch(source_image_sha256):
                    raise ValueError(f"{location}: source_image_sha256 must be lowercase SHA-256")

            if require_leakage_metadata and font_domain is not None:
                if content_group_id is None or source_image_sha256 is None:
                    raise ValueError(
                        f"{location}: supervised publication requires content_group_id and "
                        "source_image_sha256"
                    )

            if source_image_sha256 is not None:
                prior_source_hash = source_hash_bindings.setdefault(
                    source_image_sha256, (source_group_id, split, font_domain)
                )
                if prior_source_hash != (source_group_id, split, font_domain):
                    raise ValueError(
                        f"{location}: source_image_sha256 is reused across source group, split, "
                        "or font domain"
                    )

            prior_group = group_bindings.setdefault(source_group_id, (split, font_domain))
            if prior_group != (split, font_domain):
                raise ValueError(
                    f"{location}: source_group_id {source_group_id!r} crosses split or font domain"
                )

            raw_regions = raw.get("regions")
            if not isinstance(raw_regions, list) or len(raw_regions) < minimum_regions:
                raise ValueError(
                    f"{location}: regions must be an array containing at least {minimum_regions} entries"
                )
            if len(raw_regions) > maximum_regions_per_document:
                raise ValueError(
                    f"{location}: regions exceeds the {maximum_regions_per_document}-entry limit"
                )
            regions: list[FontDomainRegion] = []
            region_ids: set[str] = set()
            for region_index, raw_region in enumerate(raw_regions):
                region_location = f"{location}:regions[{region_index}]"
                if not isinstance(raw_region, Mapping):
                    raise ValueError(f"{region_location}: region must be an object")
                unknown_region_fields = sorted(set(raw_region) - _REGION_FIELDS)
                if unknown_region_fields:
                    raise ValueError(
                        f"{region_location}: unknown region fields: {', '.join(unknown_region_fields)}"
                    )
                region_id = _require_identifier(
                    raw_region.get("id"), description=f"region id at {region_location}"
                )
                if region_id in region_ids:
                    raise ValueError(f"{region_location}: duplicate region id {region_id!r}")
                role = _require_string(raw_region.get("role"), description=f"role at {region_location}")
                relative_image, image_path = _safe_image_path(
                    dataset_root, raw_region.get("image"), location=region_location
                )
                image_key = image_path.as_posix().casefold()
                if image_key in image_paths:
                    raise ValueError(f"{region_location}: region image is reused by another record")
                include = raw_region.get("include_in_consistency", role != "status_bar")
                if not isinstance(include, bool):
                    raise ValueError(f"{region_location}: include_in_consistency must be boolean")
                text = raw_region.get("text")
                if text is not None:
                    text = _require_string(text, description=f"text at {region_location}")

                image_bytes = _read_bounded_bytes(
                    image_path,
                    maximum_bytes=maximum_image_bytes,
                    description="region image",
                )
                raw_sha256 = _sha256_bytes(image_bytes)
                expected_raw = raw_region.get("raw_sha256")
                if expected_raw is not None and expected_raw != raw_sha256:
                    raise ValueError(f"{region_location}: raw_sha256 differs from the image bytes")
                rgb = _load_upright_rgb(
                    image_path,
                    image_bytes,
                    maximum_pixels=maximum_image_pixels,
                )
                pixel_sha256 = _pixel_sha256(rgb)
                expected_pixels = raw_region.get("pixel_sha256")
                if expected_pixels is not None and expected_pixels != pixel_sha256:
                    raise ValueError(f"{region_location}: pixel_sha256 differs from decoded RGB")
                perceptual_hash = _perceptual_hash(rgb)
                prior_pixels = pixel_bindings.get(pixel_sha256)
                if prior_pixels is not None:
                    if prior_pixels[1] != split:
                        raise ValueError(
                            f"{region_location}: exact decoded-pixel duplicate crosses "
                            f"{prior_pixels[1]}/{split} splits"
                        )
                    if prior_pixels[2] != font_domain:
                        raise ValueError(
                            f"{region_location}: exact decoded-pixel duplicate has conflicting "
                            "font domains"
                        )
                    if prior_pixels[0] != source_group_id:
                        raise ValueError(
                            f"{region_location}: exact decoded-pixel duplicate is assigned to "
                            "different source groups"
                        )
                pixel_bindings.setdefault(
                    pixel_sha256, (source_group_id, split, font_domain, image_key)
                )

                regions.append(
                    FontDomainRegion(
                        region_id=region_id,
                        role=role,
                        relative_image=relative_image,
                        image_path=image_path,
                        include_in_consistency=include,
                        text=text,
                        raw_sha256=raw_sha256,
                        pixel_sha256=pixel_sha256,
                        perceptual_hash=perceptual_hash,
                        width=int(rgb.shape[1]),
                        height=int(rgb.shape[0]),
                    )
                )
                region_ids.add(region_id)
                image_paths.add(image_key)

            if not any(region.include_in_consistency for region in regions):
                raise ValueError(f"{location}: at least one region must participate in consistency")
            documents.append(
                FontDomainDocument(
                    document_id=document_id,
                    source_group_id=source_group_id,
                    content_group_id=content_group_id,
                    split=split,
                    font_domain=font_domain,
                    label_source=label_source,
                    device_prior_domain=device_prior,
                    source_image_sha256=source_image_sha256,
                    regions=tuple(regions),
                )
            )
            document_ids.add(document_id)

    if not documents:
        raise ValueError(f"font-domain manifest contains no records: {manifest_path}")
    return FontDomainDataset(
        root=dataset_root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        documents=tuple(sorted(documents, key=lambda value: value.document_id)),
    )


def audit_near_duplicate_splits(
    dataset: FontDomainDataset,
    *,
    maximum_hamming_distance: int = 8,
    maximum_regions: int = 5000,
) -> dict[str, object]:
    """Fail if a small dataset contains cross-split pHash near duplicates.

    This exact pairwise audit is intentionally bounded.  Large publications
    should use an indexed near-duplicate service rather than silently skipping
    the check.
    """

    if not 0 <= maximum_hamming_distance <= 64:
        raise ValueError("maximum_hamming_distance must be between 0 and 64")
    regions = [
        (document, region)
        for document in dataset.documents
        for region in document.regions
        if document.split != "inference"
    ]
    if len(regions) > maximum_regions:
        raise ValueError(
            f"near-duplicate audit is bounded to {maximum_regions} regions; found {len(regions)}"
        )
    comparisons = 0
    for left_index, (left_document, left_region) in enumerate(regions):
        left_hash = int(left_region.perceptual_hash, 16)
        for right_document, right_region in regions[left_index + 1 :]:
            if left_document.split == right_document.split:
                continue
            comparisons += 1
            distance = (left_hash ^ int(right_region.perceptual_hash, 16)).bit_count()
            if distance <= maximum_hamming_distance:
                raise ValueError(
                    "perceptual near-duplicate crosses splits: "
                    f"{left_document.document_id}/{left_region.region_id} ({left_document.split}) vs "
                    f"{right_document.document_id}/{right_region.region_id} ({right_document.split}), "
                    f"distance={distance}"
                )
    return {
        "checked_regions": len(regions),
        "cross_split_comparisons": comparisons,
        "maximum_hamming_distance": maximum_hamming_distance,
        "perceptual_hash_abi": PERCEPTUAL_HASH_ABI,
        "passed": True,
    }


def classifier_records(dataset: FontDomainDataset) -> list[dict[str, object]]:
    """Flatten labeled, included regions for the existing CNN training path."""

    records = [
        region.classifier_record(document)
        for document in dataset.documents
        if document.split != "inference"
        for region in document.regions
        if region.include_in_consistency
    ]
    if not records:
        raise ValueError("dataset has no labeled included regions")
    return sorted(records, key=lambda value: str(value["id"]))


def _atomic_write_bytes_no_clobber(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # Some platforms/filesystems do not permit fsync on directories.
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_classifier_manifest(dataset: FontDomainDataset, output_path: Path) -> dict[str, object]:
    rows = classifier_records(dataset)
    encoded = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    output_path = Path(os.path.abspath(os.fspath(output_path.expanduser())))
    if output_path.parent.resolve() != dataset.root:
        raise ValueError(
            "classifier manifest must be written beside the source manifest so its safe relative "
            "image paths remain valid"
        )
    _atomic_write_bytes_no_clobber(output_path, encoded)
    return {
        "path": output_path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "records": len(rows),
    }
