"""Attest legacy v12/v13 sidecars across a compatibility-only exporter change.

The original guarded v13 run predates commit 17bc8af, which added two defaulted
recipient configuration fields and repeated the selected legacy backbone in
the recipient artifact metadata.  Those additions do not change the legacy
ONNX graph, but they do change JSON bytes and the contract's derived labels
hash.  This helper accepts exactly that one documented drift and rejects every
other JSON-path or value difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LEGACY_BACKBONE = "legacy_depthwise_gru_v1"
COMPATIBILITY_COMMIT = "17bc8afca6f0a1a95b0f3a45d603d016638fbbdb"
POLICY = "legacy_recipient_sidecar_defaults_added_by_17bc8af_v1"
_MISSING = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _pointer(path: tuple[str | int, ...]) -> str:
    if not path:
        return ""
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in path
    )


def _json_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _differences(
    existing: object,
    fresh: object,
    *,
    path: tuple[str | int, ...] = (),
) -> list[dict[str, object]]:
    if isinstance(existing, Mapping) and isinstance(fresh, Mapping):
        result: list[dict[str, object]] = []
        for key in sorted(set(existing) | set(fresh), key=str):
            result.extend(
                _differences(
                    existing.get(key, _MISSING),
                    fresh.get(key, _MISSING),
                    path=(*path, str(key)),
                )
            )
        return result
    if (
        isinstance(existing, Sequence)
        and not isinstance(existing, (str, bytes, bytearray))
        and isinstance(fresh, Sequence)
        and not isinstance(fresh, (str, bytes, bytearray))
    ):
        result = []
        for index in range(max(len(existing), len(fresh))):
            result.extend(
                _differences(
                    existing[index] if index < len(existing) else _MISSING,
                    fresh[index] if index < len(fresh) else _MISSING,
                    path=(*path, index),
                )
            )
        return result
    if existing is not _MISSING and fresh is not _MISSING and _json_equal(existing, fresh):
        return []
    return [
        {
            "path": _pointer(path),
            "existing_present": existing is not _MISSING,
            "fresh_present": fresh is not _MISSING,
            "existing": None if existing is _MISSING else existing,
            "fresh": None if fresh is _MISSING else fresh,
        }
    ]


def _require_contract_bindings(*, model: Path, labels: Path, contract: Mapping[str, Any]) -> None:
    if contract.get("onnx_file") != model.name:
        raise ValueError(f"Contract ONNX filename does not bind {model}")
    if contract.get("onnx_sha256") != _sha256(model):
        raise ValueError(f"Contract ONNX hash does not bind {model}")
    if contract.get("labels_file") != labels.name:
        raise ValueError(f"Contract labels filename does not bind {labels}")
    if contract.get("labels_sha256") != _sha256(labels):
        raise ValueError(f"Contract labels hash does not bind {labels}")


def _require_added_default(
    difference: Mapping[str, object], *, path: str, expected: object
) -> None:
    if difference.get("path") != path:
        raise AssertionError("caller passed a difference for the wrong path")
    if difference.get("existing_present") is not False or difference.get("fresh_present") is not True:
        raise ValueError(f"{path}: compatibility field must be fresh-only")
    if not _json_equal(difference.get("fresh"), expected):
        raise ValueError(f"{path}: compatibility field has an unsafe value")


def attest_pair(*, description: str, existing_model: Path, fresh_model: Path) -> dict[str, object]:
    existing_model = existing_model.resolve()
    fresh_model = fresh_model.resolve()
    for path in (existing_model, fresh_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing_labels = existing_model.with_suffix(".labels.json")
    fresh_labels = fresh_model.with_suffix(".labels.json")
    existing_contract = existing_model.with_suffix(".contract.json")
    fresh_contract = fresh_model.with_suffix(".contract.json")
    for path in (existing_labels, fresh_labels, existing_contract, fresh_contract):
        if not path.is_file():
            raise FileNotFoundError(path)

    existing_model_sha256 = _sha256(existing_model)
    fresh_model_sha256 = _sha256(fresh_model)
    if existing_model_sha256 != fresh_model_sha256:
        raise ValueError(f"{description}: deterministic ONNX re-export is not byte-identical")

    existing_labels_json = _load_json(existing_labels)
    fresh_labels_json = _load_json(fresh_labels)
    existing_contract_json = _load_json(existing_contract)
    fresh_contract_json = _load_json(fresh_contract)
    _require_contract_bindings(
        model=existing_model, labels=existing_labels, contract=existing_contract_json
    )
    _require_contract_bindings(model=fresh_model, labels=fresh_labels, contract=fresh_contract_json)

    labels_differences = _differences(existing_labels_json, fresh_labels_json)
    contract_differences = _differences(existing_contract_json, fresh_contract_json)
    labels_by_path = {str(item["path"]): item for item in labels_differences}
    contract_by_path = {str(item["path"]): item for item in contract_differences}
    if len(labels_by_path) != len(labels_differences) or len(contract_by_path) != len(contract_differences):
        raise AssertionError("JSON difference paths must be unique")

    expected_label_paths = {"/recipient_backbone"}
    expected_contract_paths = {
        "/labels_sha256",
        "/model/recipient_backbone",
        "/model/recipient_open_text_dropout",
        "/recipient_backbone",
    }
    observed_label_paths = set(labels_by_path)
    observed_contract_paths = set(contract_by_path)
    existing_labels_sha256 = _sha256(existing_labels)
    fresh_labels_sha256 = _sha256(fresh_labels)
    existing_contract_sha256 = _sha256(existing_contract)
    fresh_contract_sha256 = _sha256(fresh_contract)
    sidecars_byte_identical = (
        existing_labels_sha256 == fresh_labels_sha256
        and existing_contract_sha256 == fresh_contract_sha256
    )
    exact_compatibility_drift = (
        observed_label_paths == expected_label_paths
        and observed_contract_paths == expected_contract_paths
    )
    if not sidecars_byte_identical and not exact_compatibility_drift:
        raise ValueError(
            f"{description}: sidecars must be byte-identical or have the exact compatibility drift; "
            f"labels={sorted(observed_label_paths)}, contract={sorted(observed_contract_paths)}"
        )
    if sidecars_byte_identical and (labels_differences or contract_differences):
        raise AssertionError("Byte-identical sidecars cannot have JSON differences")

    if exact_compatibility_drift:
        _require_added_default(
            labels_by_path["/recipient_backbone"],
            path="/recipient_backbone",
            expected=LEGACY_BACKBONE,
        )
        _require_added_default(
            contract_by_path["/recipient_backbone"],
            path="/recipient_backbone",
            expected=LEGACY_BACKBONE,
        )
        _require_added_default(
            contract_by_path["/model/recipient_backbone"],
            path="/model/recipient_backbone",
            expected=LEGACY_BACKBONE,
        )
        _require_added_default(
            contract_by_path["/model/recipient_open_text_dropout"],
            path="/model/recipient_open_text_dropout",
            expected=0.0,
        )

    labels_hash_difference = contract_by_path.get("/labels_sha256")
    if labels_hash_difference is not None:
        if (
            labels_hash_difference.get("existing_present") is not True
            or labels_hash_difference.get("fresh_present") is not True
            or labels_hash_difference.get("existing") != existing_labels_sha256
            or labels_hash_difference.get("fresh") != fresh_labels_sha256
            or existing_labels_sha256 == fresh_labels_sha256
        ):
            raise ValueError(f"{description}: labels_sha256 drift is not derived from the two bound labels")
    elif existing_labels_sha256 != fresh_labels_sha256:
        raise ValueError(f"{description}: labels changed without the contract labels hash changing")

    # The label metadata addition and its derived contract hash must move
    # together. This prevents accepting a contract-only hash substitution.
    if ("/recipient_backbone" in labels_by_path) != (labels_hash_difference is not None):
        raise ValueError(f"{description}: label metadata and derived labels hash drift are inconsistent")

    observed_paths = {
        "labels": sorted(labels_by_path),
        "contract": sorted(contract_by_path),
    }
    return {
        "description": description,
        "passed": True,
        "onnx_byte_identical": True,
        "sidecars_byte_identical": sidecars_byte_identical,
        "existing_onnx_sha256": existing_model_sha256,
        "fresh_onnx_sha256": fresh_model_sha256,
        "existing_labels_sha256": existing_labels_sha256,
        "fresh_labels_sha256": fresh_labels_sha256,
        "existing_contract_sha256": existing_contract_sha256,
        "fresh_contract_sha256": fresh_contract_sha256,
        "observed_difference_paths": observed_paths,
        "difference_details": {
            "labels": labels_differences,
            "contract": contract_differences,
        },
    }


def attest(
    *,
    existing_seed_model: Path,
    fresh_seed_model: Path,
    existing_candidate_model: Path,
    fresh_candidate_model: Path,
) -> dict[str, object]:
    comparisons = {
        "seed": attest_pair(
            description="v12 seed",
            existing_model=existing_seed_model,
            fresh_model=fresh_seed_model,
        ),
        "candidate": attest_pair(
            description="v13 candidate",
            existing_model=existing_candidate_model,
            fresh_model=fresh_candidate_model,
        ),
    }
    return {
        "schema_version": 1,
        "kind": "receipt_unified_v13_recovery_sidecar_attestation_v1",
        "passed": True,
        "policy": POLICY,
        "compatibility_commit": COMPATIBILITY_COMMIT,
        "allowed_fresh_only_defaults": {
            "labels": {"/recipient_backbone": LEGACY_BACKBONE},
            "contract": {
                "/recipient_backbone": LEGACY_BACKBONE,
                "/model/recipient_backbone": LEGACY_BACKBONE,
                "/model/recipient_open_text_dropout": 0.0,
            },
        },
        "allowed_derived_differences": {
            "contract": ["/labels_sha256"],
            "constraint": "each contract must bind its own labels; labels metadata and hash move together",
        },
        "comparisons": comparisons,
        "all_onnx_byte_identical": all(
            bool(item["onnx_byte_identical"]) for item in comparisons.values()
        ),
        "all_sidecars_byte_identical": all(
            bool(item["sidecars_byte_identical"]) for item in comparisons.values()
        ),
        "all_sidecars_semantically_equivalent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-seed-model", type=Path, required=True)
    parser.add_argument("--fresh-seed-model", type=Path, required=True)
    parser.add_argument("--existing-candidate-model", type=Path, required=True)
    parser.add_argument("--fresh-candidate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite attestation output: {args.output}")
    result = attest(
        existing_seed_model=args.existing_seed_model,
        fresh_seed_model=args.fresh_seed_model,
        existing_candidate_model=args.existing_candidate_model,
        fresh_candidate_model=args.fresh_candidate_model,
    )
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse temporary output: {temporary}")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
