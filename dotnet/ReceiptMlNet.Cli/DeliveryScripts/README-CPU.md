# Receipt AI · Windows CPU delivery

This package runs the complete production path locally on Windows CPU:

1. receipt field detector (also produces the review boxes);
2. device type classifier;
3. unified receipt OCR for amount, time, recipient, payment method, and status;
4. full-image `max-side-1600` rectification and annotated review output.

No Python, PaddleOCR, CUDA, or network service is used at runtime. Install the
.NET 8 Desktop/Runtime x64 before verification.

## One image

Open PowerShell in this package directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-receipt-single-cpu.ps1 `
  -InputImage "D:\receipt-test\one.jpg" `
  -OutputDirectory "D:\receipt-test\one-result"
```

The command verifies every packaged file first, runs all three models on CPU,
prints the device and five candidate fields as a table, and writes the result
JSON plus an annotated image with boxes.

## A directory of images

```powershell
powershell -ExecutionPolicy Bypass -File .\run-receipt-batch-cpu.ps1 `
  -InputDirectory "D:\receipt-test\images" `
  -OutputDirectory "D:\receipt-test\batch-result"
```

Batch mode defaults to `-Annotate none` for throughput. Use `-Annotate all`
when annotated review images are required, or add `-Limit 100` for a bounded
acceptance test. The command fails if any selected image is missing or errors.

## Output policy

`candidate` values are model-recognized text for verification. Production
business values remain fail-closed as `review` until independent human-truth
calibration approves a delivery policy change. The wrappers do not weaken or
change that policy.
