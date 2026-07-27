"""Locally review OCR candidate disagreements without changing source images.

The browser UI deliberately binds only to ``127.0.0.1``.  It shows a clean
field crop together with the Paddle-derived reference and the student-model
candidate, then writes a resumable human-truth JSONL file after every decision.
No image leaves the review machine and no source/crop file is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import threading
import webbrowser
from collections import Counter
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCHEMA_VERSION = 1
KIND = "receipt_ocr_manual_review_v1"
DECISION_REFERENCE = "paddle_reference"
DECISION_CANDIDATE = "model_candidate"
DECISION_CUSTOM = "custom"
DECISION_UNCLEAR = "unclear"
REVIEW_DECISIONS = frozenset(
    (DECISION_REFERENCE, DECISION_CANDIDATE, DECISION_CUSTOM, DECISION_UNCLEAR)
)
IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".bmp", ".webp"))


class ReviewConfigurationError(ValueError):
    """Raised when an input/output review contract is unsafe or inconsistent."""


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReviewConfigurationError(f"{path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(value, Mapping):
                raise ReviewConfigurationError(f"{path}:{line_number}: expected a JSON object")
            rows.append(dict(value))
    return rows


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    return _read_jsonl(path)


def _required_text(row: Mapping[str, object], *, key: str, path: Path, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ReviewConfigurationError(f"{path}: row {index}: {key} must be a string")
    return value


def _optional_text(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) else None


def _optional_display_value(row: Mapping[str, object], key: str) -> str | None:
    """Keep numeric confidences visible for both JSONL and CSV inputs."""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _optional_provenance_text(row: Mapping[str, object], key: str) -> str | None:
    """Carry group/split provenance into human truth when the source has it."""
    value = row.get(key)
    return value if isinstance(value, str) and value else None


def _review_key(*, record_id: str, field: str, image: str) -> str:
    payload = f"{record_id}\0{field}\0{image}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_row(row: Mapping[str, object], *, path: Path, index: int) -> dict[str, object]:
    record_id = _required_text(row, key="id", path=path, index=index)
    field = _required_text(row, key="field", path=path, index=index)
    image = _required_text(row, key="image", path=path, index=index)
    reference_text = _required_text(row, key="reference_text", path=path, index=index)
    candidate_text = _required_text(row, key="candidate_text", path=path, index=index)
    if not record_id or not field or not image:
        raise ReviewConfigurationError(f"{path}: row {index}: id, field, and image must not be empty")
    input_row_sha256 = hashlib.sha256(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "review_key": _review_key(record_id=record_id, field=field, image=image),
        "input_row_sha256": input_row_sha256,
        "id": record_id,
        "field": field,
        "image": image,
        "split": _optional_provenance_text(row, "split"),
        "group_id": _optional_provenance_text(row, "group_id"),
        "source": _optional_provenance_text(row, "source"),
        "result_json": _optional_provenance_text(row, "result_json"),
        "reference_text": reference_text,
        "candidate_text": candidate_text,
        "ctc_candidate_text": _optional_text(row, "ctc_candidate_text"),
        "structured_candidate_text": _optional_text(row, "structured_candidate_text"),
        "confidence": _optional_display_value(row, "confidence"),
        "structured_confidence": _optional_display_value(row, "structured_confidence"),
        "decision": None,
        "truth_text": None,
        "notes": None,
    }


def _is_mismatch(row: Mapping[str, object]) -> bool:
    reference = row.get("reference_text")
    candidate = row.get("candidate_text")
    return isinstance(reference, str) and isinstance(candidate, str) and reference != candidate


def _load_existing_reviews(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ReviewConfigurationError(f"Review output must be a file: {path}")
    existing: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(_read_jsonl(path), start=1):
        key = raw.get("review_key")
        if not isinstance(key, str) or not key:
            raise ReviewConfigurationError(f"{path}: row {index}: missing review_key")
        if key in existing:
            raise ReviewConfigurationError(f"{path}: duplicate review_key {key}")
        existing[key] = raw
    return existing


def prepare_review_records(*, input_path: Path, output_path: Path, only_mismatches: bool = True) -> list[dict[str, object]]:
    """Create or resume a review set, preserving prior decisions by stable key."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ReviewConfigurationError("Review input and output must be different files")
    source_rows = _read_rows(input_path)
    records = [
        _normalise_row(row, path=input_path, index=index)
        for index, row in enumerate(source_rows, start=1)
        if not only_mismatches or _is_mismatch(row)
    ]
    if not records:
        raise ReviewConfigurationError("No eligible review records found")
    seen: set[str] = set()
    for record in records:
        key = str(record["review_key"])
        if key in seen:
            raise ReviewConfigurationError(f"Input contains duplicate review record: {key}")
        seen.add(key)
    existing = _load_existing_reviews(output_path)
    orphaned = sorted(set(existing) - seen)
    if orphaned:
        raise ReviewConfigurationError(
            "Existing review output contains records not present in this input. "
            "Use the same input or choose a new --output path."
        )
    for record in records:
        prior = existing.get(str(record["review_key"]))
        if prior is None:
            continue
        for key in ("decision", "truth_text", "notes"):
            value = prior.get(key)
            if value is None or isinstance(value, str):
                record[key] = value
    _atomic_write_jsonl(output_path, records)
    return records


def _validate_custom_text(value: object) -> str:
    if not isinstance(value, str):
        raise ReviewConfigurationError("custom truth_text must be a string")
    text = value.strip()
    if not text:
        raise ReviewConfigurationError("custom truth_text must not be empty")
    if any(not character.isprintable() for character in text):
        raise ReviewConfigurationError("custom truth_text contains a non-printable character")
    return text


class ReviewStore:
    """Thread-safe state that writes the entire small review set atomically."""

    def __init__(self, *, records: Sequence[Mapping[str, object]], output_path: Path) -> None:
        self._records = [dict(record) for record in records]
        self._output_path = output_path.resolve()
        self._lock = threading.RLock()

    def public_records(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(record) for record in self._records]

    def progress(self) -> dict[str, int]:
        with self._lock:
            decisions = Counter(
                str(record["decision"])
                for record in self._records
                if isinstance(record.get("decision"), str)
            )
            reviewed = sum(record.get("decision") is not None for record in self._records)
            return {
                "total": len(self._records),
                "reviewed": reviewed,
                "remaining": len(self._records) - reviewed,
                **{decision: int(decisions[decision]) for decision in sorted(REVIEW_DECISIONS)},
            }

    def image_path(self, index: int) -> Path:
        with self._lock:
            if not 0 <= index < len(self._records):
                raise IndexError(index)
            image = self._records[index]["image"]
        if not isinstance(image, str):  # Defensive: validated when records are built.
            raise ReviewConfigurationError("Review image path is invalid")
        image_path = Path(image).resolve()
        if image_path.suffix.casefold() not in IMAGE_SUFFIXES or not image_path.is_file():
            raise FileNotFoundError(image_path)
        return image_path

    def save_decision(
        self,
        *,
        index: int,
        decision: object,
        truth_text: object = None,
        notes: object = None,
    ) -> dict[str, object]:
        if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
            raise ReviewConfigurationError("decision must be paddle_reference, model_candidate, custom, or unclear")
        if notes is not None and not isinstance(notes, str):
            raise ReviewConfigurationError("notes must be a string when provided")
        with self._lock:
            if not 0 <= index < len(self._records):
                raise ReviewConfigurationError("review index is out of range")
            record = self._records[index]
            if decision == DECISION_REFERENCE:
                final_text: str | None = str(record["reference_text"])
            elif decision == DECISION_CANDIDATE:
                final_text = str(record["candidate_text"])
            elif decision == DECISION_CUSTOM:
                final_text = _validate_custom_text(truth_text)
            else:
                final_text = None
            record["decision"] = decision
            record["truth_text"] = final_text
            record["notes"] = notes.strip() if isinstance(notes, str) and notes.strip() else None
            _atomic_write_jsonl(self._output_path, self._records)
            return dict(record)


_PAGE = """<!doctype html>
<html lang=\"zh-CN\">
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Receipt OCR 人工复核</title>
<style>
  :root { color-scheme: light; font-family: system-ui, -apple-system, \"Microsoft YaHei\", sans-serif; }
  body { margin: 0; background: #f4f7fb; color: #162033; }
  main { max-width: 1180px; padding: 22px; margin: 0 auto; }
  .top { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { margin:0; font-size:24px; } #progress { color:#52607a; }
  .layout { margin-top:18px; display:grid; grid-template-columns:minmax(0, 1fr) 420px; gap:18px; }
  .card { background:#fff; border:1px solid #dce4f0; border-radius:12px; box-shadow:0 1px 2px #1b2b4812; padding:16px; }
  #imageWrap { min-height:260px; display:grid; place-items:center; background:#eef3fa; border-radius:8px; overflow:auto; }
  #crop { max-height:52vh; max-width:100%; object-fit:contain; cursor:zoom-in; image-rendering:auto; }
  #crop.zoom { max-height:none; max-width:none; cursor:zoom-out; }
  .meta { margin:12px 0 0; font-family:ui-monospace,Consolas,monospace; font-size:12px; color:#536078; overflow-wrap:anywhere; }
  .field { display:inline-block; padding:3px 9px; border-radius:20px; background:#e7efff; color:#1758b0; font-weight:650; }
  .choice { width:100%; text-align:left; margin:9px 0; padding:13px; border:2px solid #d9e0ec; border-radius:9px; background:#fff; color:#172235; font:inherit; cursor:pointer; }
  .choice:hover { border-color:#6b9bea; background:#f5f8ff; } .choice strong { display:block; margin-bottom:5px; }
  .key { display:inline-block; min-width:20px; padding:1px 4px; border-radius:4px; background:#e9edf4; font-family:ui-monospace,Consolas,monospace; text-align:center; }
  textarea { box-sizing:border-box; width:100%; min-height:74px; padding:9px; border-radius:8px; border:1px solid #bdc8da; font:inherit; }
  .small { color:#66748b; font-size:13px; line-height:1.45; } .actions { display:flex; gap:8px; margin-top:10px; } .actions button { padding:8px 11px; }
  .nav { margin-top:14px; display:flex; justify-content:space-between; gap:8px; } .nav button { padding:9px 13px; }
  #notice { min-height:22px; margin:10px 0 0; color:#a32525; } #diagnostic { white-space:pre-wrap; font-family:ui-monospace,Consolas,monospace; font-size:12px; color:#55627a; }
  @media (max-width: 850px) { main { padding:12px; } .layout { grid-template-columns:1fr; } }
</style>
<main>
  <div class=\"top\"><h1>OCR 人工复核</h1><span id=\"progress\"></span></div>
  <p class=\"small\">不修改图片。<span class=\"key\">1</span> 选 Paddle，<span class=\"key\">2</span> 选模型，<span class=\"key\">E</span> 编辑，<span class=\"key\">0</span> 标记不确定；每次选择立即保存。</p>
  <div class=\"layout\">
    <section class=\"card\"><div id=\"imageWrap\"><img id=\"crop\" alt=\"OCR crop\"></div><div class=\"meta\" id=\"meta\"></div></section>
    <section class=\"card\">
      <span class=\"field\" id=\"field\"></span>
      <h2 id=\"title\" style=\"font-size:18px\"></h2>
      <button class=\"choice\" id=\"reference\"></button>
      <button class=\"choice\" id=\"candidate\"></button>
      <label class=\"small\" for=\"custom\">自定义真实文字（按 Ctrl+Enter 保存）</label>
      <textarea id=\"custom\" placeholder=\"输入画面中真正的文字\"></textarea>
      <div class=\"actions\"><button id=\"customSave\">保存自定义</button><button id=\"unclear\">0 不确定 / 跳过</button></div>
      <p id=\"diagnostic\"></p><div id=\"notice\"></div>
      <div class=\"nav\"><button id=\"prev\">← 上一张</button><button id=\"next\">下一张 →</button></div>
    </section>
  </div>
</main>
<script>
let rows=[], current=0;
const $ = id => document.getElementById(id);
function text(v) { return v == null ? '—' : String(v); }
function nextOpen(start, direction=1) {
  if (!rows.length) return 0;
  for (let n=0;n<rows.length;n++) { const i=(start + direction*n + rows.length)%rows.length; if (!rows[i].decision) return i; }
  return Math.max(0, Math.min(start, rows.length-1));
}
function render() {
  const r=rows[current]; if (!r) return;
  $('progress').textContent=`${current+1}/${rows.length} · 已复核 ${rows.filter(x=>x.decision).length}/${rows.length}`;
  $('field').textContent=r.field; $('title').textContent=`${r.id}${r.decision ? ' · 已保存' : ''}`;
  $('meta').textContent=r.image; $('crop').src=`/api/image/${current}?v=${Date.now()}`; $('crop').classList.remove('zoom');
  $('reference').innerHTML=`<strong><span class=\"key\">1</span> Paddle / 参考标签</strong>${text(r.reference_text)}`;
  $('candidate').innerHTML=`<strong><span class=\"key\">2</span> 模型候选</strong>${text(r.candidate_text)}`;
  $('custom').value=r.decision==='custom' ? text(r.truth_text) : '';
  $('diagnostic').textContent=`CTC: ${text(r.ctc_candidate_text)}\n结构化: ${text(r.structured_candidate_text)}\n置信度: ${text(r.confidence)} / ${text(r.structured_confidence)}`;
  $('notice').textContent='';
}
async function save(decision) {
  const r=rows[current]; const body={index:current, decision, truth_text:$('custom').value};
  try {
    const response=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const result=await response.json(); if (!response.ok) throw new Error(result.error || '保存失败');
    rows[current]=result.record; current=nextOpen((current+1)%rows.length); render();
  } catch (error) { $('notice').textContent=error.message; }
}
$('reference').onclick=()=>save('paddle_reference'); $('candidate').onclick=()=>save('model_candidate');
$('customSave').onclick=()=>save('custom'); $('unclear').onclick=()=>save('unclear');
$('prev').onclick=()=>{current=(current-1+rows.length)%rows.length;render()}; $('next').onclick=()=>{current=(current+1)%rows.length;render()};
$('crop').onclick=()=>$('crop').classList.toggle('zoom');
document.addEventListener('keydown', event => {
  if (event.target.tagName==='TEXTAREA') { if (event.ctrlKey && event.key==='Enter') { event.preventDefault(); save('custom'); } return; }
  if (event.key==='1') save('paddle_reference'); else if (event.key==='2') save('model_candidate'); else if (event.key==='0') save('unclear');
  else if (event.key.toLowerCase()==='e') $('custom').focus(); else if (event.key==='ArrowLeft' || event.key.toLowerCase()==='b') $('prev').click(); else if (event.key==='ArrowRight' || event.key.toLowerCase()==='n') $('next').click();
});
fetch('/api/records').then(r=>r.json()).then(data=>{rows=data.records;current=nextOpen(0);render()}).catch(error=>{$('notice').textContent=error.message});
</script>
</html>"""


class _ReviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: ReviewStore) -> None:
        super().__init__(address, _ReviewRequestHandler)
        self.store = store


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    server: _ReviewHttpServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - match BaseHTTPRequestHandler API.
        del format, args  # Keep a keyboard review terminal quiet.

    def _send(self, *, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, *, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        self._send(
            status=status,
            body=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status=status, payload={"error": message})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        path = urlparse(self.path).path
        if path == "/":
            self._send(status=HTTPStatus.OK, body=_PAGE.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path == "/api/records":
            self._send_json(status=HTTPStatus.OK, payload={"records": self.server.store.public_records()})
            return
        if path == "/api/progress":
            self._send_json(status=HTTPStatus.OK, payload=self.server.store.progress())
            return
        if path.startswith("/api/image/"):
            try:
                index = int(path.rsplit("/", 1)[1])
                image_path = self.server.store.image_path(index)
                payload = image_path.read_bytes()
            except (ValueError, IndexError, FileNotFoundError, OSError):
                self._error(HTTPStatus.NOT_FOUND, "Review crop image was not found")
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self._send(status=HTTPStatus.OK, body=payload, content_type=content_type)
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
        if urlparse(self.path).path != "/api/decision":
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_000_000:
                raise ReviewConfigurationError("Request body is invalid")
            raw: Any = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise ReviewConfigurationError("Request body must be a JSON object")
            index = raw.get("index")
            if not isinstance(index, int):
                raise ReviewConfigurationError("review index must be an integer")
            record = self.server.store.save_decision(
                index=index,
                decision=raw.get("decision"),
                truth_text=raw.get("truth_text"),
                notes=raw.get("notes"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ReviewConfigurationError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(
            status=HTTPStatus.OK,
            payload={"record": record, "progress": self.server.store.progress()},
        )


def run_review_server(
    *,
    input_path: Path,
    output_path: Path,
    only_mismatches: bool = True,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Start the local review page until Ctrl+C, with safe resumable writes."""
    if host != "127.0.0.1":
        raise ReviewConfigurationError("For privacy, the review UI may bind only to 127.0.0.1")
    if not 0 <= port <= 65535:
        raise ReviewConfigurationError("port must be in [0, 65535]")
    records = prepare_review_records(
        input_path=input_path,
        output_path=output_path,
        only_mismatches=only_mismatches,
    )
    server = _ReviewHttpServer((host, port), ReviewStore(records=records, output_path=output_path))
    address, chosen_port = server.server_address[:2]
    url = f"http://{address}:{chosen_port}/"
    print(f"OCR review UI: {url}")
    print(f"Resumable labels: {output_path.resolve()}")
    print("Keys: 1=Paddle reference, 2=model candidate, E=edit, 0=unclear, B/N=previous/next. Ctrl+C saves and exits.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped; all submitted decisions are already saved.")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review OCR Paddle/model disagreements in an offline localhost browser page")
    parser.add_argument("--input", type=Path, required=True, help="comparisons.jsonl or CSV with id,field,reference_text,candidate_text,image")
    parser.add_argument("--output", type=Path, required=True, help="New/resumable reviewed truth JSONL")
    parser.add_argument("--all", dest="only_mismatches", action="store_false", help="Review every input row, not only reference/candidate mismatches")
    parser.set_defaults(only_mismatches=True)
    parser.add_argument("--port", type=int, default=8765, help="Local port; use 0 to choose a free port")
    parser.add_argument("--no-open", dest="open_browser", action="store_false", help="Do not automatically open the local browser")
    parser.set_defaults(open_browser=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run_review_server(
            input_path=args.input,
            output_path=args.output,
            only_mismatches=args.only_mismatches,
            port=args.port,
            open_browser=args.open_browser,
        )
    except (OSError, ReviewConfigurationError, ValueError) as error:
        raise SystemExit(f"OCR review failed:\n{error}") from None


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()
