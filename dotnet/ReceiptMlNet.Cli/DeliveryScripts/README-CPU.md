# Receipt AI · Windows CPU delivery

This package runs the complete production path locally on Windows CPU:

1. receipt field detector (also produces the review boxes);
2. device type classifier;
3. unified receipt OCR for amount, time, recipient, payment method, and status;
4. teacher-compatible portrait orientation (landscape rotates 90° clockwise), full-image `max-side-1600` rectification, and annotated review output.

No Python, PaddleOCR, CUDA, or network service is used at runtime. Install the
.NET 8 Runtime x64 (`Microsoft.NETCore.App 8.x`) before verification. The .NET 8
Desktop Runtime also satisfies this prerequisite.

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
contract/labels sidecar, runs all three models on CPU, and rejects the result
unless all model filenames and hashes match those artifacts. It then prints the
device and five candidate fields as a table and writes the result JSON plus
annotated images with boxes.

## A directory of images

```powershell
powershell -ExecutionPolicy Bypass -File .\run-receipt-batch-cpu.ps1 `
  -InputDirectory "D:\receipt-test\images" `
  -OutputDirectory "D:\receipt-test\batch-result"
```

Batch mode defaults to `-Annotate none` for throughput. Use `-Annotate all`
when annotated review images are required, or add `-Limit 100` for a bounded
acceptance test. The command fails if any selected image is missing or errors,
or if its manifest, summary, CPU provider, and error evidence disagree.

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
