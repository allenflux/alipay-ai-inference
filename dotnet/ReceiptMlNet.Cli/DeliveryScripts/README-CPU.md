# Receipt AI · Windows CPU delivery

This package runs the complete production path locally on Windows CPU:

1. receipt field detector (also produces the review boxes);
2. device type classifier;
3. architecture-v13 unified receipt OCR for amount, time, payment method, and visible transfer status;
4. recipient-only PP-OCR detection, angle classification, and recognition;
5. teacher-compatible portrait orientation (landscape rotates 90° clockwise), full-image `max-side-1600` rectification, and annotated review output.

Every model in this path runs as pure ONNX through the .NET CPU runtime.
No Python, Paddle framework, CUDA, or network service is used in production. The
recipient PP-OCR model family is delivered as verified ONNX files with its
exact dictionary and contract. Install the .NET 8 Runtime x64
(`Microsoft.NETCore.App 8.x`) before verification. The .NET 8 Desktop Runtime
also satisfies this prerequisite.

The production entrypoints require an architecture-v13 package, whose
`status_text_logits` output reads the visible transfer-status text with CTC.
They reject legacy v12 packages because v12 only has a status
classifier and therefore cannot honestly label its status candidate as OCR.
There is no production override for this check; use the base CLI directly for
explicit legacy diagnostics.

## One image

Open PowerShell in this package directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-receipt-single-cpu.ps1 `
  -InputImage "D:\receipt-test\one.jpg" `
  -OutputDirectory "D:\receipt-test\one-result"
```

The command verifies that the package hash manifest exactly covers every
payload file (the manifest itself is the sole exclusion), checks the accepted
full-validation CPU evidence, verifies every delivered model and adjacent
contract/labels sidecar, and independently verifies the contained recipient
PP-OCR contract, dictionary, model hashes, and byte sizes. It runs detector,
device classifier, v13 OCR, and recipient PP-OCR on CPU and rejects the result
unless both ONNX providers report `cpu` and every result fingerprint matches
the delivered artifacts. It then prints the device and candidate fields,
followed by transfer status `Raw OCR`, `Normalized`, and `Review state`, and
writes the result JSON plus annotated images with boxes.

The embedded release evidence is also checked at startup. It must be an
accepted 10,016-image full-validation CPU A/B run from the exact delivered CLI
build, with 100% result and candidate coverage. The fixed exact-match floors
are amount 78.85%, time 98.40%, payment method 93.25%, recipient 90%, and
visible transfer status 90%; non-success-to-success errors must remain zero and
the hybrid CPU p95 overhead ceiling may not exceed 250 ms. The package's fresh
end-to-end 10,016-image score is checked independently against the same five
accuracy, coverage, and transfer-status safety guards; passing A/B evidence
cannot mask a regression during final package validation.

The A/B latency claim is recalculated from the two contained, hash-bound CPU
runtime summaries. The complete `app/` publish closure is also enumerated and
matched against its canonically sorted path/hash/size manifest, so a missing,
changed, or extra managed/native runtime file is rejected before inference.

## A directory of images

```powershell
powershell -ExecutionPolicy Bypass -File .\run-receipt-batch-cpu.ps1 `
  -InputDirectory "D:\receipt-test\images" `
  -OutputDirectory "D:\receipt-test\batch-result"
```

Batch mode defaults to `-Annotate none` for throughput. Use `-Annotate all`
when annotated review images are required, or add `-Limit 100` for a bounded
acceptance test. The command fails if any selected image is missing or errors,
or if its manifest, summary, either CPU provider, and error evidence disagree.

Output directories must be new and outside this immutable package. For batch
mode the output directory must also be outside, and not contain, the input
directory; this prevents generated JPG files from contaminating later inputs.
The package, input tree, output paths, and their existing ancestors must not use
reparse points, Windows device namespaces, substituted DOS drives, drive-relative
aliases, reserved DOS device names, or alternate data streams. Ordinary physical
absolute paths such as `D:\receipt-test\images` remain supported. Batch `-Limit`
validation is bound to the CLI's Ordinal path sort and exact first-N selection.

## Output policy

`candidate` values are model-recognized text for verification. Production
business values remain fail-closed as `review` until independent human-truth
calibration approves a delivery policy change. The wrappers do not weaken or
change that policy. Before reporting PASS they inspect amount, time, recipient,
and payment method independently: every candidate must retain state/value/
delivery value `review`, while a missing candidate may only be `absent` or
`unreadable` and may not expose a different business value.
