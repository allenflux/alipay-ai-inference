from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from transfer_receipt_ai.ocr_train import (
    GENERIC_TEXT_LINE_FIELD,
    RecognizerConfig,
    build_train_parser,
    build_ctc_recognizer,
    decode_ctc_logits,
    export_onnx,
    train_recognizer,
)


def _record(image: str, text: str, split: str, *, group_id: str | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": image,
        "image": image,
        "field": "amount",
        "text": text,
        "split": split,
        "group_id": group_id or image,
    }


def test_tiny_ctc_training_writes_a_checkpoint(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for index, (text, split) in enumerate((("1", "train"), ("2", "train"), ("1", "val"), ("2", "val"))):
        image_name = f"images/{index}.png"
        pixels = np.full((20, 50, 3), 255, dtype=np.uint8)
        pixels[:, 5 + index * 3 : 12 + index * 3] = 30
        Image.fromarray(pixels).save(dataset / image_name)
        records.append(_record(image_name, text, split))
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    checkpoint = train_recognizer(
        records_path=records_path,
        output_dir=tmp_path / "run",
        fields=("amount",),
        config=RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1),
        device="cpu",
        epochs=1,
        batch_size=2,
    )

    assert checkpoint.is_file()
    assert (checkpoint.parent / "last.pt").is_file()
    assert (checkpoint.parent / "charset.json").is_file()


def test_training_defaults_include_recipient_field() -> None:
    args = build_train_parser().parse_args(["--records", "records.jsonl", "--output", "out"])
    assert "recipient_field" in args.fields


def test_training_parser_accepts_generic_line_only_and_rejects_mixed_mode() -> None:
    args = build_train_parser().parse_args(
        ["--records", "generic_text_lines.jsonl", "--output", "out", "--fields", GENERIC_TEXT_LINE_FIELD]
    )
    assert args.fields == (GENERIC_TEXT_LINE_FIELD,)
    with pytest.raises(SystemExit):
        build_train_parser().parse_args(
            [
                "--records",
                "generic_text_lines.jsonl",
                "--output",
                "out",
                "--fields",
                f"amount,{GENERIC_TEXT_LINE_FIELD}",
            ]
        )


def test_ctc_decoder_collapses_repeats_and_blanks() -> None:
    # CTC classes: blank=0, A=1, B=2.  The index sequence is A,A,blank,B,B,A.
    logits = np.zeros((6, 1, 3), dtype=np.float32)
    for time, index in enumerate((1, 1, 0, 2, 2, 1)):
        logits[time, 0, index] = 1.0
    assert decode_ctc_logits(logits, characters=["A", "B"]) == ["ABA"]


def test_training_rejects_validation_characters_not_seen_in_train(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(2):
        Image.fromarray(np.full((20, 50, 3), 255, dtype=np.uint8)).save(images / f"{index}.png")
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                _record("images/0.png", "1", "train"),
                _record("images/1.png", "2", "val"),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from the training charset"):
        train_recognizer(
            records_path=records_path,
            output_dir=tmp_path / "run",
            fields=("amount",),
            config=RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1),
            device="cpu",
            epochs=1,
            batch_size=1,
        )


def test_training_keeps_test_characters_out_of_the_training_charset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(3):
        Image.fromarray(np.full((20, 50, 3), 255, dtype=np.uint8)).save(images / f"{index}.png")
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                _record("images/0.png", "1", "train"),
                _record("images/1.png", "1", "val"),
                _record("images/2.png", "不在训练集", "test"),
            )
        ),
        encoding="utf-8",
    )

    checkpoint = train_recognizer(
        records_path=records_path,
        output_dir=tmp_path / "run",
        fields=("amount",),
        config=RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1),
        device="cpu",
        epochs=1,
        batch_size=1,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["characters"] == ["1"]


def test_training_rejects_ctc_targets_that_cannot_fit_time_axis(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(2):
        Image.fromarray(np.full((20, 50, 3), 255, dtype=np.uint8)).save(images / f"{index}.png")
    # Input width 64 produces 16 CTC time steps. Seventeen repeated digits
    # require 33 steps because CTC must put blanks between equal neighbours.
    too_long = "1" * 17
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                _record("images/0.png", too_long, "train"),
                _record("images/1.png", too_long, "val"),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot fit the recognizer CTC time axis"):
        train_recognizer(
            records_path=records_path,
            output_dir=tmp_path / "run",
            fields=("amount",),
            config=RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1),
            device="cpu",
            epochs=1,
            batch_size=1,
        )


def test_training_rejects_a_group_split_across_train_and_validation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(2):
        Image.fromarray(np.full((20, 50, 3), 255, dtype=np.uint8)).save(images / f"{index}.png")
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                _record("images/0.png", "1", "train", group_id="same-receipt"),
                _record("images/1.png", "1", "val", group_id="same-receipt"),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appears in both train and val splits"):
        train_recognizer(
            records_path=records_path,
            output_dir=tmp_path / "run",
            fields=("amount",),
            config=RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1),
            device="cpu",
            epochs=1,
            batch_size=1,
        )


def test_training_rejects_requested_field_without_train_or_validation_coverage(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(3):
        Image.fromarray(np.full((20, 50, 3), 255, dtype=np.uint8)).save(images / f"{index}.png")
    records_path = dataset / "pseudo_labels.jsonl"
    records = (
        _record("images/0.png", "1", "train"),
        _record("images/1.png", "1", "val"),
        {**_record("images/2.png", "1", "train"), "field": "time"},
    )
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="No validation samples remain.*time"):
        train_recognizer(
            records_path=records_path,
            output_dir=tmp_path / "run",
            fields=("amount", "time"),
            config=RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1),
            device="cpu",
            epochs=1,
            batch_size=1,
        )


def test_onnx_export_matches_torch_when_onnx_dependencies_are_installed(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for index, (text, split) in enumerate((("1", "train"), ("2", "train"), ("1", "val"), ("2", "val"))):
        image_name = f"images/{index}.png"
        pixels = np.full((20, 50, 3), 255, dtype=np.uint8)
        pixels[:, 4 + index * 4 : 12 + index * 4] = 25
        Image.fromarray(pixels).save(dataset / image_name)
        records.append(_record(image_name, text, split))
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    config = RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1)
    checkpoint = train_recognizer(
        records_path=records_path,
        output_dir=tmp_path / "run",
        fields=("amount",),
        config=config,
        device="cpu",
        epochs=1,
        batch_size=2,
    )

    onnx_path, charset_path, contract_path = export_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "receipt_ocr_ctc_v1.onnx",
    )
    onnx.checker.check_model(onnx.load_model(onnx_path))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_ctc_recognizer(vocab_size=len(payload["characters"]) + 1, config=config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    sample = torch.rand((1, 1, config.image_height, config.image_width), dtype=torch.float32)
    expected = model(sample).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": sample.numpy()})[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert charset_path.is_file()
    assert contract_path.is_file()
