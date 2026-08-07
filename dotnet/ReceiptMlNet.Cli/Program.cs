using System.Diagnostics;
using System.Globalization;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Transforms.Onnx;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

return await ReceiptMlNetProgram.RunAsync(args);

internal static class ReceiptMlNetProgram
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true,
    };

    public static Task<int> RunAsync(string[] args)
    {
        try
        {
            var options = CliOptions.Parse(args);
            Run(options);
            return Task.FromResult(0);
        }
        catch (UsageException exception)
        {
            Console.Error.WriteLine(exception.Message);
            Console.Error.WriteLine();
            Console.Error.WriteLine(CliOptions.Usage);
            return Task.FromResult(2);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"ML.NET inference failed: {exception.Message}");
            return Task.FromResult(1);
        }
    }

    private static void Run(CliOptions options)
    {
        var totalStopwatch = Stopwatch.StartNew();
        var detectorContract = ModelContract.LoadAndVerify(options.DetectorPath, "receipt_lrcnn_v1");
        ModelContract? deviceContract = options.DeviceModelPath is null
            ? null
            : ModelContract.LoadAndVerify(options.DeviceModelPath, "statusbar_device_v1");
        var ocrBundle = options.OcrMode == "onnx"
            ? PaddleOcrDeliveryBundle.LoadAndVerify(options.OcrBundlePath!)
            : null;
        var unifiedOcrBundle = options.OcrMode == "unified"
            ? UnifiedOcrBundle.LoadAndVerify(options.OcrModelPath!)
            : null;

        var inputFiles = options.InputListPath is null
            ? EnumerateInputFiles(options.InputPath!).ToList()
            : ReadInputList(options.InputListPath).ToList();
        if (inputFiles.Count == 0)
        {
            var inputDescription = options.InputListPath ?? options.InputPath;
            throw new UsageException($"No supported image files found under {inputDescription}");
        }
        if (options.Limit is not null)
        {
            inputFiles = inputFiles.Take(options.Limit.Value).ToList();
        }

        Directory.CreateDirectory(options.OutputDirectory);
        var sourceRoot = options.InputPath is null
            ? null
            : File.Exists(options.InputPath)
                ? Path.GetDirectoryName(Path.GetFullPath(options.InputPath))!
                : Path.GetFullPath(options.InputPath);
        var workItems = inputFiles
            .Select(inputFile => new InputWorkItem(
                inputFile,
                options.InputListPath is null
                    ? OutputPathFor(options.OutputDirectory, sourceRoot!, inputFile)
                    : OutputPathForInputList(options.OutputDirectory, inputFile)))
            .ToList();
        var outputPathComparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        if (workItems.Select(item => item.Output).Distinct(outputPathComparer).Count() != workItems.Count)
        {
            throw new UsageException("Input paths produced duplicate output names; refusing to overwrite a result");
        }
        var manifest = new List<ManifestRecord>();
        var inferenceLatencies = new List<double>();
        var stageLatencies = new List<InferenceStageLatency>();
        var errorCount = 0;
        Exception? fatalException = null;
        var errorsPath = Path.Combine(options.OutputDirectory, "inference_errors.jsonl");
        File.WriteAllText(errorsPath, string.Empty, Encoding.UTF8);

        var device = DeviceSetting.Parse(options.Device);
        // The inference loop below is intentionally serial. Keep one mutable
        // detector tensor and its pinned OrtValue per Run invocation rather
        // than allocating/copying roughly 15.2 MiB for every receipt; never
        // share this buffer across workers.
        var detectorInputBuffer = new float[ImagePipeline.DetectorTensorLength];
        using var detector = new DetectorModel(
            options.DetectorPath,
            device,
            options.DetectorIntraOpThreads,
            detectorInputBuffer);
        var deviceClassifier = options.DeviceModelPath is null
            ? null
            : new DeviceModel(options.DeviceModelPath, device);
        Console.WriteLine($"Requested ONNX device: {device.Requested} (receipt detector{(deviceClassifier is null ? string.Empty : "/device model")})");
        Console.WriteLine($"Detector intra-op threads: {(options.DetectorIntraOpThreads?.ToString(CultureInfo.InvariantCulture) ?? "runtime default")}");
        using var ocrEngine = ocrBundle is null ? null : new PaddleOcrEngine(ocrBundle, device);
        using var unifiedOcrEngine = unifiedOcrBundle is null ? null : new UnifiedOcrEngine(unifiedOcrBundle, device);
        if (ocrEngine is not null)
        {
            Console.WriteLine($"OCR ONNX execution provider: {ocrEngine.ExecutionProvider} (det/cls/rec)");
        }
        if (unifiedOcrEngine is not null)
        {
            Console.WriteLine($"Unified OCR ONNX execution provider: {unifiedOcrEngine.ExecutionProvider} (one v12 session/run per receipt)");
        }

        foreach (var workItem in workItems)
        {
            var inputFile = workItem.Source;
            var outputFile = workItem.Output;
            var annotationPaths = AnnotationPaths.ForResultJson(outputFile);
            if (options.SkipExisting && ExistingResultSatisfiesRequestedMode(
                    outputFile,
                    detectorContract,
                    deviceContract,
                    ocrBundle,
                    unifiedOcrBundle,
                    options.Rectification))
            {
                manifest.Add(new ManifestRecord(
                    Path.GetFullPath(inputFile),
                    outputFile,
                    "skipped_existing",
                    annotationPaths.Rectified,
                    annotationPaths.Original));
                continue;
            }

            try
            {
                var inferenceStopwatch = Stopwatch.StartNew();
                var result = InferImage(
                    inputFile,
                    detector,
                    detectorInputBuffer,
                    deviceClassifier,
                    ocrEngine,
                    unifiedOcrEngine,
                    options.ScoreThreshold,
                    options.Rectification,
                    out var stageLatency);
                inferenceStopwatch.Stop();
                var inferenceMs = Math.Round(inferenceStopwatch.Elapsed.TotalMilliseconds, 4);
                if (options.RequireComplete)
                {
                    EnsureCoreFields(result.Detections);
                }
                result = result with
                {
                    ModelContracts = new ContractReferences(
                        Detector: detectorContract.FileName,
                        Device: deviceContract?.FileName,
                        DetectorSha256: detectorContract.ModelSha256,
                        DetectorContractSha256: detectorContract.ContractSha256,
                        DeviceSha256: deviceContract?.ModelSha256,
                        DeviceContractSha256: deviceContract?.ContractSha256,
                        OcrBundle: ocrBundle is null ? null : Path.GetFileName(ocrBundle.ContractPath),
                        OcrBundleContractSha256: ocrBundle?.ContractSha256,
                        UnifiedOcrModel: unifiedOcrBundle is null ? null : Path.GetFileName(unifiedOcrBundle.ModelPath),
                        UnifiedOcrContract: unifiedOcrBundle is null ? null : Path.GetFileName(unifiedOcrBundle.ContractPath),
                        UnifiedOcrModelSha256: unifiedOcrBundle?.ModelSha256,
                        UnifiedOcrLabelsSha256: unifiedOcrBundle?.LabelsSha256,
                        UnifiedOcrContractSha256: unifiedOcrBundle?.ContractSha256),
                };
                if (ShouldAnnotate(result.Detections, options.AnnotationMode))
                {
                    AnnotationRenderer.RenderAndSave(inputFile, result.Detections, result.Device, annotationPaths);
                }
                WriteJsonAtomic(outputFile, result);
                manifest.Add(new ManifestRecord(
                    Path.GetFullPath(inputFile),
                    outputFile,
                    "written",
                    annotationPaths.Rectified,
                    annotationPaths.Original,
                    inferenceMs,
                    stageLatency));
                inferenceLatencies.Add(inferenceMs);
                stageLatencies.Add(stageLatency);
            }
            catch (Exception exception)
            {
                errorCount++;
                var error = new ErrorRecord(Path.GetFullPath(inputFile), exception.GetType().Name, exception.Message);
                File.AppendAllText(errorsPath, JsonSerializer.Serialize(error, JsonOptions) + Environment.NewLine, Encoding.UTF8);
                if (!options.ContinueOnError)
                {
                    fatalException = exception;
                    break;
                }
            }
        }

        WriteJsonAtomic(Path.Combine(options.OutputDirectory, "inference_manifest.json"), manifest);
        totalStopwatch.Stop();
        var writtenCount = manifest.Count(record => record.Status == "written");
        var skippedCount = manifest.Count(record => record.Status == "skipped_existing");
        var summary = new InferenceSummary(
            device.Requested,
            unifiedOcrEngine?.ExecutionProvider,
            options.DetectorIntraOpThreads,
            workItems.Count,
            writtenCount,
            skippedCount,
            errorCount,
            Math.Round(totalStopwatch.Elapsed.TotalSeconds, 4),
            SummarizeLatencies(inferenceLatencies),
            SummarizeStageLatencies(stageLatencies));
        WriteJsonAtomic(Path.Combine(options.OutputDirectory, "inference_summary.json"), summary);
        if (fatalException is not null)
        {
            ExceptionDispatchInfo.Capture(fatalException).Throw();
        }
        Console.WriteLine($"Wrote {writtenCount} ML.NET result bundle(s) to {options.OutputDirectory}");
    }

    private static ReceiptResult InferImage(
        string inputFile,
        DetectorModel detector,
        float[] detectorInputBuffer,
        DeviceModel? deviceClassifier,
        PaddleOcrEngine? ocrEngine,
        UnifiedOcrEngine? unifiedOcrEngine,
        float scoreThreshold,
        string rectificationMode,
        out InferenceStageLatency stageLatency)
    {
        var stageStopwatch = Stopwatch.StartNew();
        using var source = ImagePipeline.LoadUprightRgb(inputFile);
        var imageLoadMs = StopAndReadMilliseconds(stageStopwatch);

        double? deviceMs = null;
        stageStopwatch.Restart();
        var device = deviceClassifier?.Classify(source);
        if (deviceClassifier is not null)
        {
            deviceMs = StopAndReadMilliseconds(stageStopwatch);
        }

        stageStopwatch.Restart();
        using var rectification = ReceiptRectifier.Rectify(source, rectificationMode);
        var rectified = rectification.Image;
        var prepared = ImagePipeline.PrepareDetectorInput(rectified, detectorInputBuffer);
        var detectorPreprocessMs = StopAndReadMilliseconds(stageStopwatch);

        stageStopwatch.Restart();
        var predictions = detector.Predict(prepared.Tensor);
        var detectorInferenceMs = StopAndReadMilliseconds(stageStopwatch);

        stageStopwatch.Restart();
        var detections = PostProcessDetections(predictions, prepared, scoreThreshold);
        var detectorPostprocessMs = StopAndReadMilliseconds(stageStopwatch);

        double? paddleOcrMs = null;
        if (ocrEngine is not null)
        {
            stageStopwatch.Restart();
            detections = EnrichWithOcr(rectified, detections, ocrEngine);
            paddleOcrMs = StopAndReadMilliseconds(stageStopwatch);
        }

        UnifiedOcrStageLatency? unifiedOcrLatency = null;
        UnifiedOcrReadResult? unifiedOcr = null;
        if (unifiedOcrEngine is not null)
        {
            unifiedOcr = unifiedOcrEngine.RecognizeReceipt(rectified, detections, out var measuredUnifiedOcrLatency);
            unifiedOcrLatency = measuredUnifiedOcrLatency;
        }

        stageStopwatch.Restart();
        if (unifiedOcr is not null)
        {
            detections = EnrichWithUnifiedOcr(detections, unifiedOcr);
        }
        var fields = ocrEngine is not null
            ? BuildFields(detections)
            : unifiedOcr is not null
                ? BuildUnifiedFields(detections, unifiedOcr)
                : null;
        // Detector and OCR boxes remain in rectified coordinates through all
        // crop operations.  Only the public bbox_image values are projected
        // back to the EXIF-upright source coordinate system.
        var outputDetections = detections
            .Select(detection => detection with { BboxImage = rectification.ProjectBoxToSource(detection.BboxImage) })
            .ToList();
        var resultAssemblyMs = StopAndReadMilliseconds(stageStopwatch);

        stageLatency = new InferenceStageLatency(
            imageLoadMs,
            deviceMs,
            detectorPreprocessMs,
            detectorInferenceMs,
            detectorPostprocessMs,
            paddleOcrMs,
            unifiedOcrLatency?.PreprocessMs,
            unifiedOcrLatency?.InferenceMs,
            unifiedOcrLatency?.PostprocessMs,
            resultAssemblyMs);

        var rectificationGeometry = rectification.Geometry();
        return new ReceiptResult(
            Path.GetFullPath(inputFile),
            "mlnet",
            new DetectorGeometry(
                rectificationGeometry.SourceSize,
                rectificationGeometry.RectifiedSize,
                new ImageSize(ImagePipeline.DetectorWidth, ImagePipeline.DetectorHeight),
                "letterbox",
                rectificationGeometry.Rectification,
                rectificationGeometry.RotationDegrees,
                rectificationGeometry.ScreenDetected,
                rectificationGeometry.ScreenQuadOriginal,
                rectificationGeometry.HOriginalToRectified,
                rectificationGeometry.HRectifiedToOriginal),
            outputDetections,
            fields,
            device,
            null,
            new[]
            {
                "This .NET CLI performs ONNX model inference.",
                "EXIF orientation is applied before inference. Rectification mode max-side-1600 rotates landscape inputs 90 degrees clockwise, mirrors Python's full-image warp, and does not detect or crop a phone/screen boundary.",
                "Perspective photos still require an externally rectified input; automatic screen detection is intentionally outside this production mode.",
                "bbox_image and both compatibility-named annotated JPGs use EXIF-upright source coordinates.",
                ocrEngine is not null
                    ? "OCR uses the verified PP-OCR ONNX delivery bundle on the selected rectified image."
                    : unifiedOcrEngine is not null
                        ? "OCR uses the verified architecture-v12 unified ONNX reader in one session/run per receipt. Its current text/status contract is review-only: candidates are diagnostic and delivered values fail closed to review."
                        : "OCR field extraction is disabled; use --ocr onnx --ocr-bundle <delivery-directory> or --ocr unified --ocr-model <v12-reader.onnx> to enable it.",
            });
    }

    private static double StopAndReadMilliseconds(Stopwatch stopwatch)
    {
        stopwatch.Stop();
        return Math.Round(stopwatch.Elapsed.TotalMilliseconds, 4);
    }

    private static List<DetectionResult> PostProcessDetections(
        DetectorOutput output,
        DetectorInputTensor prepared,
        float scoreThreshold)
    {
        var count = Math.Min(output.Labels.Length, Math.Min(output.Scores.Length, output.Boxes.Length / 4));
        var bestByLabel = new Dictionary<string, DetectionResult>(StringComparer.Ordinal);
        for (var index = 0; index < count; index++)
        {
            var score = output.Scores[index];
            if (!float.IsFinite(score) || score < scoreThreshold || !Labels.TryGetValue(output.Labels[index], out var label))
            {
                continue;
            }

            var boxOffset = index * 4;
            var restored = prepared.RestoreBox(
                output.Boxes[boxOffset],
                output.Boxes[boxOffset + 1],
                output.Boxes[boxOffset + 2],
                output.Boxes[boxOffset + 3]);
            var candidate = new DetectionResult(label, score, restored);
            if (!bestByLabel.TryGetValue(label, out var previous) || candidate.Score > previous.Score)
            {
                bestByLabel[label] = candidate;
            }
        }
        return bestByLabel.Values.OrderBy(item => item.Label, StringComparer.Ordinal).ToList();
    }

    private static List<DetectionResult> EnrichWithOcr(
        Image<Rgb24> source,
        IReadOnlyList<DetectionResult> detections,
        PaddleOcrEngine ocrEngine)
    {
        var enriched = new List<DetectionResult>(detections.Count);
        foreach (var detection in detections)
        {
            var crop = UnifiedOcrImageOps.CropFieldWithMargin(source, detection.BboxImage);
            if (crop is null)
            {
                enriched.Add(detection with { Ocr = new OcrResult(string.Empty, null) });
                continue;
            }
            using (crop)
            {
                var result = ocrEngine.Recognize(crop);
                enriched.Add(detection with { Ocr = new OcrResult(result.Text, result.Confidence) });
            }
        }
        return enriched;
    }

    private static ReceiptFields BuildFields(IReadOnlyList<DetectionResult> detections)
    {
        var byLabel = detections.ToDictionary(item => item.Label, StringComparer.Ordinal);
        var time = FieldFromOcr(byLabel.GetValueOrDefault("time"));
        if (time.Raw is not null)
        {
            time = time with { Value = ReceiptFieldNormalizer.NormalizeTime(time.Raw) };
        }

        var amount = FieldFromOcr(byLabel.GetValueOrDefault("amount"));
        if (amount.Raw is not null && ReceiptFieldNormalizer.NormalizeAmount(amount.Raw) is { } normalizedAmount)
        {
            amount = amount with
            {
                Normalized = normalizedAmount.Normalized,
                AmountFen = normalizedAmount.AmountFen,
                Currency = normalizedAmount.Currency,
            };
        }

        var recipient = FieldFromOcr(byLabel.GetValueOrDefault("recipient_field"));
        if (recipient.Raw is not null)
        {
            recipient = recipient with { Value = ReceiptFieldNormalizer.ExtractFieldValue(recipient.Raw, "recipient") };
        }

        var paymentMethod = FieldFromOcr(byLabel.GetValueOrDefault("payment_method_field"));
        if (paymentMethod.Raw is not null)
        {
            var value = ReceiptFieldNormalizer.ExtractFieldValue(paymentMethod.Raw, "payment_method");
            paymentMethod = paymentMethod with
            {
                Value = value,
                Normalized = ReceiptFieldNormalizer.NormalizePaymentMethod(value).Normalized,
            };
        }

        var transferStatus = FieldFromOcr(byLabel.GetValueOrDefault("transfer_status"));
        if (transferStatus.Raw is not null)
        {
            transferStatus = transferStatus with { Normalized = ReceiptFieldNormalizer.NormalizeStatus(transferStatus.Raw) };
        }

        return new ReceiptFields(time, amount, transferStatus, recipient, paymentMethod);
    }

    private static ReceiptFieldResult FieldFromOcr(DetectionResult? detection)
    {
        if (detection is null)
        {
            return new ReceiptFieldResult("absent", null, null, null, null, null, null, null, null);
        }
        if (detection.Ocr is null || string.IsNullOrWhiteSpace(detection.Ocr.Text))
        {
            // Python's unreadable-field shape exposes the detector value as
            // `score`, while successful OCR uses `detector_score`.
            return new ReceiptFieldResult("unreadable", null, null, null, MathF.Round(detection.Score, 6), null, null, null, null);
        }
        return new ReceiptFieldResult(
            "read",
            detection.Ocr.Text,
            detection.Ocr.Confidence is null ? null : MathF.Round(detection.Ocr.Confidence.Value, 6),
            MathF.Round(detection.Score, 6),
            null,
            null,
            null,
            null,
            null);
    }

    private static List<DetectionResult> EnrichWithUnifiedOcr(
        IReadOnlyList<DetectionResult> detections,
        UnifiedOcrReadResult unifiedOcr)
    {
        var enriched = new List<DetectionResult>(detections.Count);
        foreach (var detection in detections)
        {
            var candidate = detection.Label == "transfer_status"
                ? unifiedOcr.StatusCandidate is null
                    ? null
                    : new OcrResult(unifiedOcr.StatusCandidate, unifiedOcr.StatusConfidence)
                : unifiedOcr.Candidates.TryGetValue(detection.Label, out var fieldCandidate)
                    ? new OcrResult(fieldCandidate.Candidate, fieldCandidate.Confidence)
                    : null;
            enriched.Add(detection with { Ocr = candidate ?? new OcrResult(string.Empty, null) });
        }
        return enriched;
    }

    /// <summary>
    /// Keep v12 candidate diagnostics in the JSON, but do not promote
    /// Paddle-derived pseudo-label text into a business value.  The persisted
    /// v12 contract decides delivery policy, which is currently review-only.
    /// </summary>
    private static ReceiptFields BuildUnifiedFields(
        IReadOnlyList<DetectionResult> detections,
        UnifiedOcrReadResult unifiedOcr)
    {
        var byLabel = detections.ToDictionary(item => item.Label, StringComparer.Ordinal);
        return new ReceiptFields(
            UnifiedTextField(
                byLabel.GetValueOrDefault("time"),
                unifiedOcr.Candidates.GetValueOrDefault("time"),
                unifiedOcr.TextRuntimePolicy,
                unifiedOcr.TextDeliveryValue),
            UnifiedTextField(
                byLabel.GetValueOrDefault("amount"),
                unifiedOcr.Candidates.GetValueOrDefault("amount"),
                unifiedOcr.TextRuntimePolicy,
                unifiedOcr.TextDeliveryValue),
            UnifiedStatusField(byLabel.GetValueOrDefault("transfer_status"), unifiedOcr),
            UnifiedTextField(
                byLabel.GetValueOrDefault("recipient_field"),
                unifiedOcr.Candidates.GetValueOrDefault("recipient_field"),
                unifiedOcr.TextRuntimePolicy,
                unifiedOcr.TextDeliveryValue),
            UnifiedTextField(
                byLabel.GetValueOrDefault("payment_method_field"),
                unifiedOcr.Candidates.GetValueOrDefault("payment_method_field"),
                unifiedOcr.TextRuntimePolicy,
                unifiedOcr.TextDeliveryValue));
    }

    private static ReceiptFieldResult UnifiedTextField(
        DetectionResult? detection,
        UnifiedOcrCandidate? candidate,
        string policy,
        string deliveryValue)
    {
        if (detection is null)
        {
            return new ReceiptFieldResult("absent", null, null, null, null, null, null, null, null, DeliveryPolicy: policy);
        }
        if (candidate is null || string.IsNullOrWhiteSpace(candidate.Candidate))
        {
            return new ReceiptFieldResult(
                "unreadable",
                null,
                null,
                null,
                MathF.Round(detection.Score, 6),
                deliveryValue,
                null,
                null,
                null,
                DeliveryPolicy: policy,
                DeliveryValue: deliveryValue);
        }
        return new ReceiptFieldResult(
            deliveryValue == "review" ? "review" : "read",
            candidate.Candidate,
            MathF.Round(candidate.Confidence, 6),
            MathF.Round(detection.Score, 6),
            null,
            candidate.DeliveryValue,
            null,
            null,
            null,
            Candidate: candidate.Candidate,
            CtcCandidate: candidate.CtcCandidate,
            CtcConfidence: MathF.Round(candidate.CtcConfidence, 6),
            StructuredCandidate: candidate.StructuredCandidate,
            StructuredConfidence: candidate.StructuredConfidence is null ? null : MathF.Round(candidate.StructuredConfidence.Value, 6),
            DeliveryPolicy: policy,
            DeliveryValue: candidate.DeliveryValue);
    }

    private static ReceiptFieldResult UnifiedStatusField(DetectionResult? detection, UnifiedOcrReadResult unifiedOcr)
    {
        if (detection is null)
        {
            return new ReceiptFieldResult("absent", null, null, null, null, null, null, null, null, DeliveryPolicy: unifiedOcr.StatusRuntimePolicy);
        }
        if (string.IsNullOrWhiteSpace(unifiedOcr.StatusCandidate))
        {
            return new ReceiptFieldResult(
                "unreadable",
                null,
                null,
                null,
                MathF.Round(detection.Score, 6),
                unifiedOcr.StatusDeliveryValue,
                null,
                null,
                null,
                DeliveryPolicy: unifiedOcr.StatusRuntimePolicy,
                DeliveryValue: unifiedOcr.StatusDeliveryValue);
        }
        return new ReceiptFieldResult(
            unifiedOcr.StatusDeliveryValue == "review" ? "review" : "read",
            unifiedOcr.StatusCandidate,
            unifiedOcr.StatusConfidence is null ? null : MathF.Round(unifiedOcr.StatusConfidence.Value, 6),
            MathF.Round(detection.Score, 6),
            null,
            unifiedOcr.StatusDeliveryValue,
            unifiedOcr.StatusRuntimePolicy == "classify" ? unifiedOcr.StatusCandidate : null,
            null,
            null,
            Candidate: unifiedOcr.StatusCandidate,
            DeliveryPolicy: unifiedOcr.StatusRuntimePolicy,
            DeliveryValue: unifiedOcr.StatusDeliveryValue);
    }

    private static bool ExistingResultSatisfiesRequestedMode(
        string outputPath,
        ModelContract detector,
        ModelContract? device,
        PaddleOcrDeliveryBundle? paddleOcr,
        UnifiedOcrBundle? unifiedOcr,
        string rectificationMode)
    {
        if (!File.Exists(outputPath))
        {
            return false;
        }
        // Re-run unless the existing JSON proves that every current model
        // artifact is byte-identical.  This prevents --skip-existing from
        // silently retaining detector/device/OCR output after a same-name
        // ONNX, labels, contract, or policy swap.
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(outputPath));
            var requiresOcrFields = paddleOcr is not null || unifiedOcr is not null;
            if ((requiresOcrFields
                    && (!document.RootElement.TryGetProperty("fields", out var fields)
                        || fields.ValueKind != JsonValueKind.Object))
                || !document.RootElement.TryGetProperty("geometry", out var geometry)
                || geometry.ValueKind != JsonValueKind.Object
                || !HasJsonString(geometry, "rectification", rectificationMode)
                || !HasCurrentRectificationGeometry(geometry, rectificationMode)
                || !document.RootElement.TryGetProperty("model_contracts", out var contracts)
                || contracts.ValueKind != JsonValueKind.Object)
            {
                return false;
            }
            return HasJsonString(contracts, "detector", detector.FileName)
                && HasJsonString(contracts, "detector_sha256", detector.ModelSha256)
                && HasJsonString(contracts, "detector_contract_sha256", detector.ContractSha256)
                && HasOptionalJsonString(contracts, "device", device?.FileName)
                && HasOptionalJsonString(contracts, "device_sha256", device?.ModelSha256)
                && HasOptionalJsonString(contracts, "device_contract_sha256", device?.ContractSha256)
                && HasOptionalJsonString(
                    contracts,
                    "ocr_bundle",
                    paddleOcr is null ? null : Path.GetFileName(paddleOcr.ContractPath))
                && HasOptionalJsonString(
                    contracts,
                    "ocr_bundle_contract_sha256",
                    paddleOcr?.ContractSha256)
                && HasOptionalJsonString(
                    contracts,
                    "unified_ocr_model",
                    unifiedOcr is null ? null : Path.GetFileName(unifiedOcr.ModelPath))
                && HasOptionalJsonString(
                    contracts,
                    "unified_ocr_contract",
                    unifiedOcr is null ? null : Path.GetFileName(unifiedOcr.ContractPath))
                && HasOptionalJsonString(
                    contracts,
                    "unified_ocr_model_sha256",
                    unifiedOcr?.ModelSha256)
                && HasOptionalJsonString(
                    contracts,
                    "unified_ocr_labels_sha256",
                    unifiedOcr?.LabelsSha256)
                && HasOptionalJsonString(
                    contracts,
                    "unified_ocr_contract_sha256",
                    unifiedOcr?.ContractSha256);
        }
        catch (JsonException)
        {
            return false;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static bool HasJsonString(JsonElement source, string propertyName, string expected)
    {
        return source.TryGetProperty(propertyName, out var property)
            && property.ValueKind == JsonValueKind.String
            && string.Equals(property.GetString(), expected, StringComparison.OrdinalIgnoreCase);
    }

    private static bool HasOptionalJsonString(JsonElement source, string propertyName, string? expected)
    {
        if (expected is null)
        {
            return !source.TryGetProperty(propertyName, out var property)
                || property.ValueKind == JsonValueKind.Null;
        }
        return HasJsonString(source, propertyName, expected);
    }

    private static bool HasCurrentRectificationGeometry(JsonElement geometry, string rectificationMode)
    {
        if (rectificationMode == ReceiptRectifier.NoneMode)
        {
            return true;
        }
        if (rectificationMode != ReceiptRectifier.MaxSide1600Mode
            || !geometry.TryGetProperty("source_size", out var sourceSize)
            || sourceSize.ValueKind != JsonValueKind.Object
            || !sourceSize.TryGetProperty("width", out var sourceWidthProperty)
            || !sourceWidthProperty.TryGetInt32(out var sourceWidth)
            || !sourceSize.TryGetProperty("height", out var sourceHeightProperty)
            || !sourceHeightProperty.TryGetInt32(out var sourceHeight)
            || sourceWidth < 2
            || sourceHeight < 2
            || !geometry.TryGetProperty("rectified_size", out var rectifiedSize)
            || rectifiedSize.ValueKind != JsonValueKind.Object
            || !rectifiedSize.TryGetProperty("width", out var rectifiedWidthProperty)
            || !rectifiedWidthProperty.TryGetInt32(out var rectifiedWidth)
            || !rectifiedSize.TryGetProperty("height", out var rectifiedHeightProperty)
            || !rectifiedHeightProperty.TryGetInt32(out var rectifiedHeight)
            || !geometry.TryGetProperty("rotation_degrees", out var rotationProperty)
            || !rotationProperty.TryGetInt32(out var rotationDegrees)
            || !geometry.TryGetProperty("screen_detected", out var screenDetectedProperty)
            || screenDetectedProperty.ValueKind != JsonValueKind.False)
        {
            return false;
        }
        // Invalidate old landscape outputs that used the same mode string but
        // did not apply the teacher pipeline's deterministic portrait rule.
        var expectedRotation = sourceWidth > sourceHeight ? 90 : 0;
        var expectedWidth = expectedRotation == 90 ? sourceHeight : sourceWidth;
        var expectedHeight = expectedRotation == 90 ? sourceWidth : sourceHeight;
        var longestSide = Math.Max(expectedWidth, expectedHeight);
        if (longestSide > ReceiptRectifier.MaximumSide)
        {
            var scale = (double)ReceiptRectifier.MaximumSide / longestSide;
            expectedWidth = Math.Max(
                2,
                (int)Math.Round(expectedWidth * scale, MidpointRounding.ToEven));
            expectedHeight = Math.Max(
                2,
                (int)Math.Round(expectedHeight * scale, MidpointRounding.ToEven));
        }
        return rotationDegrees == expectedRotation
            && rectifiedWidth == expectedWidth
            && rectifiedHeight == expectedHeight;
    }

    private static void EnsureCoreFields(IEnumerable<DetectionResult> detections)
    {
        var found = detections.Select(item => item.Label).ToHashSet(StringComparer.Ordinal);
        var missing = RequiredLabels.Where(label => !found.Contains(label)).ToArray();
        if (missing.Length > 0)
        {
            throw new InvalidOperationException($"incomplete detection: missing required transfer fields; missing={string.Join(',', missing)}");
        }
    }

    private static bool ShouldAnnotate(IEnumerable<DetectionResult> detections, string mode)
    {
        return mode switch
        {
            "none" => false,
            "flagged" => RequiredLabels.Except(detections.Select(item => item.Label), StringComparer.Ordinal).Any(),
            _ => true,
        };
    }

    private static IEnumerable<string> EnumerateInputFiles(string inputPath)
    {
        var fullPath = Path.GetFullPath(inputPath);
        if (File.Exists(fullPath))
        {
            if (!ImageExtensions.Contains(Path.GetExtension(fullPath)))
            {
                throw new UsageException($"Unsupported image extension: {fullPath}");
            }
            return new[] { fullPath };
        }
        if (!Directory.Exists(fullPath))
        {
            throw new UsageException($"Input path does not exist: {fullPath}");
        }
        return Directory.EnumerateFiles(fullPath, "*", SearchOption.AllDirectories)
            .Where(path => ImageExtensions.Contains(Path.GetExtension(path)))
            .OrderBy(path => path, StringComparer.Ordinal);
    }

    private static IReadOnlyList<string> ReadInputList(string inputListPath)
    {
        var fullListPath = Path.GetFullPath(inputListPath);
        if (!File.Exists(fullListPath))
        {
            throw new UsageException($"Input list does not exist: {fullListPath}");
        }

        var listDirectory = Path.GetDirectoryName(fullListPath)!;
        var pathComparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        var seen = new HashSet<string>(pathComparer);
        var resolved = new List<string>();
        var lineNumber = 0;
        foreach (var rawLine in File.ReadLines(fullListPath))
        {
            lineNumber++;
            var entry = rawLine.Trim();
            if (entry.Length == 0 || entry.StartsWith('#'))
            {
                continue;
            }

            string fullPath;
            try
            {
                fullPath = Path.GetFullPath(entry, listDirectory);
            }
            catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException)
            {
                throw new UsageException($"Invalid image path at {fullListPath}:{lineNumber}: {exception.Message}");
            }
            if (!File.Exists(fullPath))
            {
                throw new UsageException($"Input image at {fullListPath}:{lineNumber} does not exist: {fullPath}");
            }
            if (!ImageExtensions.Contains(Path.GetExtension(fullPath)))
            {
                throw new UsageException($"Unsupported image extension at {fullListPath}:{lineNumber}: {fullPath}");
            }
            if (seen.Add(fullPath))
            {
                resolved.Add(fullPath);
            }
        }

        if (resolved.Count == 0)
        {
            throw new UsageException($"Input list contains no image paths: {fullListPath}");
        }
        return resolved;
    }

    private static string OutputPathFor(string outputDirectory, string sourceRoot, string sourcePath)
    {
        var relative = Path.GetRelativePath(sourceRoot, sourcePath);
        if (relative.StartsWith("..", StringComparison.Ordinal))
        {
            relative = Path.GetFileName(sourcePath);
        }
        return Path.Combine(outputDirectory, Path.ChangeExtension(relative, ".json"));
    }

    private static string OutputPathForInputList(string outputDirectory, string sourcePath)
    {
        var identity = Path.GetFullPath(sourcePath);
        if (OperatingSystem.IsWindows())
        {
            identity = identity.ToUpperInvariant();
        }
        var digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(identity))).ToLowerInvariant();
        return Path.Combine(outputDirectory, "input-list", digest + ".json");
    }

    private static LatencySummary SummarizeLatencies(IReadOnlyCollection<double> latencies)
    {
        if (latencies.Count == 0)
        {
            return new LatencySummary(0, null, null, null);
        }
        var sorted = latencies.OrderBy(value => value).ToArray();
        return new LatencySummary(
            sorted.Length,
            Math.Round(sorted.Average(), 4),
            Math.Round(Percentile(sorted, 0.50), 4),
            Math.Round(Percentile(sorted, 0.95), 4));
    }

    private static InferenceStageLatencySummary SummarizeStageLatencies(
        IReadOnlyCollection<InferenceStageLatency> latencies)
    {
        return new InferenceStageLatencySummary(
            SummarizeLatencies(latencies.Select(item => item.ImageLoad).ToArray()),
            SummarizeOptionalLatencies(latencies.Select(item => item.Device)),
            SummarizeLatencies(latencies.Select(item => item.DetectorPreprocess).ToArray()),
            SummarizeLatencies(latencies.Select(item => item.DetectorInference).ToArray()),
            SummarizeLatencies(latencies.Select(item => item.DetectorPostprocess).ToArray()),
            SummarizeOptionalLatencies(latencies.Select(item => item.PaddleOcr)),
            SummarizeOptionalLatencies(latencies.Select(item => item.UnifiedOcrPreprocess)),
            SummarizeOptionalLatencies(latencies.Select(item => item.UnifiedOcrInference)),
            SummarizeOptionalLatencies(latencies.Select(item => item.UnifiedOcrPostprocess)),
            SummarizeLatencies(latencies.Select(item => item.ResultAssembly).ToArray()));
    }

    private static LatencySummary SummarizeOptionalLatencies(IEnumerable<double?> latencies)
    {
        return SummarizeLatencies(latencies.Where(value => value.HasValue).Select(value => value!.Value).ToArray());
    }

    private static double Percentile(IReadOnlyList<double> sorted, double quantile)
    {
        var position = (sorted.Count - 1) * quantile;
        var lower = (int)Math.Floor(position);
        var upper = (int)Math.Ceiling(position);
        if (lower == upper)
        {
            return sorted[lower];
        }
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static void WriteJsonAtomic<T>(string path, T payload)
    {
        var fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        var temporary = fullPath + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(payload, JsonOptions) + Environment.NewLine, Encoding.UTF8);
        File.Move(temporary, fullPath, overwrite: true);
    }

    private static readonly Dictionary<long, string> Labels = new()
    {
        [1] = "time",
        [2] = "amount",
        [3] = "transfer_status",
        [4] = "recipient_field",
        [5] = "payment_method_field",
    };

    private static readonly string[] RequiredLabels =
    {
        "time", "amount", "transfer_status", "recipient_field", "payment_method_field",
    };

    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    };
}

internal sealed class DetectorModel : IDisposable
{
    private static readonly string[] InputNames = ["image"];
    private static readonly string[] OutputNames = ["boxes", "labels", "scores"];
    private static readonly long[] InputShape = [3, ImagePipeline.DetectorHeight, ImagePipeline.DetectorWidth];
    private static readonly int[] InputMetadataShape = [3, ImagePipeline.DetectorHeight, ImagePipeline.DetectorWidth];

    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;
    private readonly OrtValue _inputValue;
    private readonly OrtValue[] _inputValues;
    private readonly float[] _inputBuffer;
    private bool _disposed;

    public DetectorModel(
        string modelPath,
        DeviceSetting device,
        int? intraOpThreads,
        float[] inputBuffer)
    {
        ArgumentNullException.ThrowIfNull(inputBuffer);
        if (inputBuffer.Length != ImagePipeline.DetectorTensorLength)
        {
            throw new ArgumentException(
                $"Detector input buffer must contain exactly {ImagePipeline.DetectorTensorLength} floats; found {inputBuffer.Length}",
                nameof(inputBuffer));
        }
        if (intraOpThreads is not null && device.GpuDeviceId is not null)
        {
            throw new ArgumentException("Detector intra-op threads are supported only for an explicit CPU session", nameof(intraOpThreads));
        }

        var session = CreateSession(modelPath, device, intraOpThreads);
        try
        {
            VerifyModelAbi(session);
            // The detector loop mutates this same fixed buffer before every
            // serial Run. OrtValue pins it once and does not copy its contents.
            var inputValue = OrtValue.CreateTensorValueFromMemory(inputBuffer, InputShape);
            OrtValue[] inputValues;
            RunOptions runOptions;
            try
            {
                inputValues = [inputValue];
                runOptions = new RunOptions();
            }
            catch
            {
                inputValue.Dispose();
                throw;
            }
            _session = session;
            _inputValue = inputValue;
            _inputValues = inputValues;
            _runOptions = runOptions;
            _inputBuffer = inputBuffer;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    public DetectorOutput Predict(float[] tensor)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DetectorModel));
        }
        if (!ReferenceEquals(tensor, _inputBuffer))
        {
            throw new ArgumentException(
                "Detector inference must use the fixed buffer pinned by its reusable OrtValue",
                nameof(tensor));
        }

        using var runtimeOutputs = _session.Run(_runOptions, InputNames, _inputValues, OutputNames);
        if (runtimeOutputs.Count != OutputNames.Length)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector returned {runtimeOutputs.Count} outputs; expected {OutputNames.Length}");
        }
        var outputs = runtimeOutputs.ToArray();
        var boxes = ReadBoxes(outputs[0]);
        var labels = ReadVector<long>(outputs[1], "labels", TensorElementType.Int64);
        var scores = ReadVector<float>(outputs[2], "scores", TensorElementType.Float);
        var boxCount = boxes.Length / 4;
        if (labels.Length != boxCount || scores.Length != boxCount)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector output lengths disagree: boxes={boxCount}, labels={labels.Length}, scores={scores.Length}");
        }
        return new DetectorOutput { Boxes = boxes, Labels = labels, Scores = scores };
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _inputValue.Dispose();
        _runOptions.Dispose();
        _session.Dispose();
    }

    private static InferenceSession CreateSession(
        string modelPath,
        DeviceSetting device,
        int? intraOpThreads)
    {
        if (device.GpuDeviceId is null)
        {
            if (intraOpThreads is null)
            {
                return new InferenceSession(modelPath);
            }
            using var options = new SessionOptions
            {
                IntraOpNumThreads = intraOpThreads.Value,
            };
            return new InferenceSession(modelPath, options);
        }

        try
        {
            using var options = new SessionOptions();
            options.AppendExecutionProvider_CUDA(device.GpuDeviceId.Value);
            return new InferenceSession(modelPath, options);
        }
        catch (OnnxRuntimeException) when (device.FallbackToCpu)
        {
            return new InferenceSession(modelPath);
        }
    }

    private static void VerifyModelAbi(InferenceSession session)
    {
        if (!HasExactNames(session.InputNames, InputNames))
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector inputs must be exactly [{string.Join(',', InputNames)}]; found [{string.Join(',', session.InputNames)}]");
        }
        if (!HasExactNames(session.OutputNames, OutputNames))
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector outputs must be exactly [{string.Join(',', OutputNames)}]; found [{string.Join(',', session.OutputNames)}]");
        }

        var input = RequireTensorMetadata(session.InputMetadata, "image", typeof(float));
        if (!input.Dimensions.SequenceEqual(InputMetadataShape))
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector input image has shape [{string.Join(',', input.Dimensions)}], expected [{string.Join(',', InputMetadataShape)}]");
        }

        var boxes = RequireTensorMetadata(session.OutputMetadata, "boxes", typeof(float));
        VerifyOutputMetadataShape(boxes.Dimensions, "boxes", rank: 2, trailingDimension: 4);
        var labels = RequireTensorMetadata(session.OutputMetadata, "labels", typeof(long));
        VerifyOutputMetadataShape(labels.Dimensions, "labels", rank: 1, trailingDimension: null);
        var scores = RequireTensorMetadata(session.OutputMetadata, "scores", typeof(float));
        VerifyOutputMetadataShape(scores.Dimensions, "scores", rank: 1, trailingDimension: null);
    }

    private static NodeMetadata RequireTensorMetadata(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string name,
        Type expectedElementType)
    {
        if (!metadata.TryGetValue(name, out var value)
            || !value.IsTensor
            || value.ElementType != expectedElementType)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector {name} must be a tensor of {expectedElementType.Name}");
        }
        return value;
    }

    private static void VerifyOutputMetadataShape(
        IReadOnlyList<int> shape,
        string name,
        int rank,
        int? trailingDimension)
    {
        var validLeadingDimension = shape.Count > 0 && (shape[0] == -1 || shape[0] > 0);
        var validTrailingDimension = trailingDimension is null || shape[^1] == trailingDimension.Value;
        if (shape.Count != rank || !validLeadingDimension || !validTrailingDimension)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector output {name} has invalid shape [{string.Join(',', shape)}]");
        }
    }

    private static float[] ReadBoxes(OrtValue value)
    {
        var typeAndShape = value.GetTensorTypeAndShape();
        var shape = typeAndShape.Shape;
        if (typeAndShape.ElementDataType != TensorElementType.Float
            || shape.Length != 2
            || shape[0] < 0
            || shape[1] != 4)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector output boxes must be a float tensor [N,4]; found {typeAndShape.ElementDataType} [{string.Join(',', shape)}]");
        }
        var data = value.GetTensorDataAsSpan<float>();
        if (typeAndShape.ElementCount != data.Length || shape[0] * 4 != data.Length)
        {
            throw new InvalidOperationException("ONNX receipt detector output boxes has inconsistent shape metadata");
        }
        return data.ToArray();
    }

    private static T[] ReadVector<T>(OrtValue value, string name, TensorElementType expectedElementType)
        where T : unmanaged
    {
        var typeAndShape = value.GetTensorTypeAndShape();
        var shape = typeAndShape.Shape;
        if (typeAndShape.ElementDataType != expectedElementType
            || shape.Length != 1
            || shape[0] < 0)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector output {name} must be a {expectedElementType} tensor [N]; found {typeAndShape.ElementDataType} [{string.Join(',', shape)}]");
        }
        var data = value.GetTensorDataAsSpan<T>();
        if (typeAndShape.ElementCount != data.Length || shape[0] != data.Length)
        {
            throw new InvalidOperationException(
                $"ONNX receipt detector output {name} has inconsistent shape metadata");
        }
        return data.ToArray();
    }

    private static bool HasExactNames(IEnumerable<string> actual, IReadOnlyCollection<string> expected)
    {
        var names = actual.ToArray();
        return names.Length == expected.Count
            && names.Distinct(StringComparer.Ordinal).Count() == names.Length
            && names.ToHashSet(StringComparer.Ordinal).SetEquals(expected);
    }

}

internal sealed class DeviceModel
{
    private readonly PredictionEngine<DeviceInput, DeviceOutput> _engine;
    // PredictionEngine is already restricted to the serial receipt loop. Keep
    // its status-bar tensor private to the same non-thread-safe model instance.
    private readonly float[] _statusbarInputBuffer = new float[ImagePipeline.StatusbarTensorLength];

    public DeviceModel(string modelPath, DeviceSetting device)
    {
        var context = new MLContext(seed: 1);
        var fitData = context.Data.LoadFromEnumerable(new[]
        {
            new DeviceInput { Statusbar = new float[ImagePipeline.StatusbarTensorLength] },
        });
        var pipeline = context.Transforms.ApplyOnnxModel(
            outputColumnNames: new[] { "probabilities" },
            inputColumnNames: new[] { "statusbar" },
            modelFile: modelPath,
            gpuDeviceId: device.GpuDeviceId,
            fallbackToCpu: device.FallbackToCpu);
        var model = pipeline.Fit(fitData);
        _engine = context.Model.CreatePredictionEngine<DeviceInput, DeviceOutput>(model);
    }

    public DeviceResult Classify(Image<Rgb24> source)
    {
        var resolutionPlatform = ResolutionPlatform(source.Width, source.Height);
        if (resolutionPlatform is "ios" or "android")
        {
            var result = new DeviceResult(resolutionPlatform, DeviceCn[resolutionPlatform], "resolution", 0.99f, false, null, null, null);
            var pIos = PredictPIos(source);
            var cnnPlatform = pIos > 0.5f ? "ios" : "android";
            var cnnConfidence = Math.Max(pIos, 1.0f - pIos);
            if (cnnPlatform != resolutionPlatform && cnnConfidence >= 0.8f)
            {
                return result with
                {
                    Confidence = 0.5f,
                    DevicePriorConflict = true,
                    CnnPlatform = cnnPlatform,
                    ConflictDetail = $"分辨率判{resolutionPlatform}、状态栏判{cnnPlatform}(疑似缩放伪造)",
                };
            }
            return result;
        }

        var probability = PredictPIos(source);
        var confidence = Math.Max(probability, 1.0f - probability);
        var platform = confidence < 0.75f ? "uncertain" : (probability > 0.5f ? "ios" : "android");
        return new DeviceResult(
            platform,
            DeviceCn[platform],
            "cnn",
            MathF.Round(confidence, 3),
            false,
            MathF.Round(probability, 4),
            null,
            null);
    }

    private float PredictPIos(Image<Rgb24> source)
    {
        var probabilities = _engine.Predict(new DeviceInput
        {
            Statusbar = ImagePipeline.PrepareStatusbarInput(source, _statusbarInputBuffer),
        }).Probabilities;
        if (probabilities.Length != 2)
        {
            throw new InvalidOperationException($"ONNX status-bar model must return two probabilities; found length {probabilities.Length}");
        }
        return probabilities[1];
    }

    private static string ResolutionPlatform(int width, int height)
    {
        if (IphoneResolutions.Contains((width, height)))
        {
            return "ios";
        }
        return AndroidPanelWidths.Contains(Math.Min(width, height)) ? "android" : "abstain";
    }

    private static readonly Dictionary<string, string> DeviceCn = new(StringComparer.Ordinal)
    {
        ["ios"] = "苹果",
        ["android"] = "安卓",
        ["uncertain"] = "不确定",
    };

    private static readonly HashSet<(int Width, int Height)> IphoneResolutions = new()
    {
        (640, 960), (640, 1136), (750, 1334), (828, 1792), (960, 640), (1125, 2436),
        (1136, 640), (1170, 2532), (1179, 2556), (1206, 2622), (1242, 2208), (1242, 2688),
        (1284, 2778), (1290, 2796), (1320, 2868), (1334, 750), (1792, 828), (2208, 1242),
        (2436, 1125), (2532, 1170), (2556, 1179), (2622, 1206), (2688, 1242), (2778, 1284),
        (2796, 1290), (2868, 1320),
    };

    private static readonly HashSet<int> AndroidPanelWidths = new() { 720, 1080, 1440 };
}

internal static class ImagePipeline
{
    public const int DetectorWidth = 864;
    public const int DetectorHeight = 1536;
    public const int DetectorTensorLength = 3 * DetectorHeight * DetectorWidth;
    public const int StatusbarWidth = 512;
    public const int StatusbarHeight = 64;
    public const int StatusbarTensorLength = 3 * StatusbarHeight * StatusbarWidth;

    public static Image<Rgb24> LoadUprightRgb(string path)
    {
        var image = Image.Load<Rgb24>(path);
        image.Mutate(context => context.AutoOrient());
        return image;
    }

    public static DetectorInputTensor PrepareDetectorInput(Image<Rgb24> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return PrepareDetectorInputCore(source, new float[DetectorTensorLength], clearLetterboxPadding: false);
    }

    /// <summary>
    /// Prepare detector input in a caller-owned tensor buffer.
    ///
    /// Every destination element is overwritten on each call: letterbox
    /// padding is cleared and the resized rectangle fills all three RGB planes.
    /// A destination must not be shared by concurrent inference operations.
    /// </summary>
    public static DetectorInputTensor PrepareDetectorInput(Image<Rgb24> source, float[] destination)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Length != DetectorTensorLength)
        {
            throw new ArgumentException(
                $"Detector tensor destination must contain exactly {DetectorTensorLength} floats; found {destination.Length}",
                nameof(destination));
        }
        return PrepareDetectorInputCore(source, destination, clearLetterboxPadding: true);
    }

    private static DetectorInputTensor PrepareDetectorInputCore(
        Image<Rgb24> source,
        float[] values,
        bool clearLetterboxPadding)
    {
        var scale = Math.Min((float)DetectorWidth / source.Width, (float)DetectorHeight / source.Height);
        var resizedWidth = Math.Clamp((int)Math.Round(source.Width * scale, MidpointRounding.ToEven), 1, DetectorWidth);
        var resizedHeight = Math.Clamp((int)Math.Round(source.Height * scale, MidpointRounding.ToEven), 1, DetectorHeight);
        var left = (DetectorWidth - resizedWidth) / 2;
        var top = (DetectorHeight - resizedHeight) / 2;
        // ImageSharp calls the bilinear kernel "Triangle".
        using var resized = source.Clone(context => context.Resize(resizedWidth, resizedHeight, KnownResamplers.Triangle));
        if (clearLetterboxPadding)
        {
            ClearDetectorLetterboxPadding(values, resizedWidth, resizedHeight, left, top);
        }

        // The padding is zero and the resized pixels overwrite the full content
        // rectangle, exactly matching the former black canvas + opaque
        // DrawImage operation.
        var plane = DetectorHeight * DetectorWidth;
        resized.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < resizedHeight; y++)
            {
                var row = accessor.GetRowSpan(y);
                var destinationRow = (y + top) * DetectorWidth + left;
                for (var x = 0; x < row.Length; x++)
                {
                    var pixel = row[x];
                    var offset = destinationRow + x;
                    values[offset] = pixel.R / 255.0f;
                    values[plane + offset] = pixel.G / 255.0f;
                    values[2 * plane + offset] = pixel.B / 255.0f;
                }
            }
        });
        return new DetectorInputTensor(values, source.Width, source.Height, (float)resizedWidth / source.Width, (float)resizedHeight / source.Height, left, top);
    }

    private static void ClearDetectorLetterboxPadding(
        float[] values,
        int resizedWidth,
        int resizedHeight,
        int left,
        int top)
    {
        var plane = DetectorHeight * DetectorWidth;
        var firstContentRow = top;
        var firstBottomPaddingRow = top + resizedHeight;
        var firstRightPaddingColumn = left + resizedWidth;
        for (var channel = 0; channel < 3; channel++)
        {
            var planeOffset = channel * plane;
            if (firstContentRow > 0)
            {
                Array.Clear(values, planeOffset, firstContentRow * DetectorWidth);
            }
            if (firstBottomPaddingRow < DetectorHeight)
            {
                Array.Clear(
                    values,
                    planeOffset + firstBottomPaddingRow * DetectorWidth,
                    (DetectorHeight - firstBottomPaddingRow) * DetectorWidth);
            }
            for (var y = firstContentRow; y < firstBottomPaddingRow; y++)
            {
                var rowOffset = planeOffset + y * DetectorWidth;
                if (left > 0)
                {
                    Array.Clear(values, rowOffset, left);
                }
                if (firstRightPaddingColumn < DetectorWidth)
                {
                    Array.Clear(
                        values,
                        rowOffset + firstRightPaddingColumn,
                        DetectorWidth - firstRightPaddingColumn);
                }
            }
        }
    }

    public static float[] PrepareStatusbarInput(Image<Rgb24> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return PrepareStatusbarInputCore(source, new float[StatusbarTensorLength]);
    }

    /// <summary>
    /// Prepare status-bar input in a caller-owned tensor buffer. Every element
    /// is overwritten on each call; do not share a destination concurrently.
    /// </summary>
    public static float[] PrepareStatusbarInput(Image<Rgb24> source, float[] destination)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Length != StatusbarTensorLength)
        {
            throw new ArgumentException(
                $"Status-bar tensor destination must contain exactly {StatusbarTensorLength} floats; found {destination.Length}",
                nameof(destination));
        }
        return PrepareStatusbarInputCore(source, destination);
    }

    private static float[] PrepareStatusbarInputCore(Image<Rgb24> source, float[] values)
    {
        var stripHeight = Math.Max(1, (int)Math.Round(source.Height * 0.08, MidpointRounding.ToEven));
        using var strip = source.Clone(context => context.Crop(new Rectangle(0, 0, source.Width, stripHeight)));
        // The Python training/inference path calls Pillow resize without an
        // explicit filter for RGB, whose default is bicubic.
        using var canvas = strip.Clone(context => context.Resize(StatusbarWidth, StatusbarHeight, KnownResamplers.Bicubic));
        var plane = StatusbarHeight * StatusbarWidth;
        canvas.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < StatusbarHeight; y++)
            {
                var row = accessor.GetRowSpan(y);
                var destinationRow = y * StatusbarWidth;
                for (var x = 0; x < row.Length; x++)
                {
                    var pixel = row[x];
                    var offset = destinationRow + x;
                    values[offset] = (pixel.R / 255.0f - 0.485f) / 0.229f;
                    values[plane + offset] = (pixel.G / 255.0f - 0.456f) / 0.224f;
                    values[2 * plane + offset] = (pixel.B / 255.0f - 0.406f) / 0.225f;
                }
            }
        });
        return values;
    }
}

internal sealed record DetectorInputTensor(
    float[] Tensor,
    int SourceWidth,
    int SourceHeight,
    float ScaleX,
    float ScaleY,
    int OffsetX,
    int OffsetY)
{
    public float[] RestoreBox(float x1, float y1, float x2, float y2)
    {
        return new[]
        {
            Math.Clamp((x1 - OffsetX) / ScaleX, 0.0f, SourceWidth),
            Math.Clamp((y1 - OffsetY) / ScaleY, 0.0f, SourceHeight),
            Math.Clamp((x2 - OffsetX) / ScaleX, 0.0f, SourceWidth),
            Math.Clamp((y2 - OffsetY) / ScaleY, 0.0f, SourceHeight),
        };
    }
}

internal sealed class DetectorOutput
{
    [ColumnName("boxes")]
    public float[] Boxes { get; set; } = Array.Empty<float>();

    [ColumnName("labels")]
    public long[] Labels { get; set; } = Array.Empty<long>();

    [ColumnName("scores")]
    public float[] Scores { get; set; } = Array.Empty<float>();
}

internal sealed class DeviceInput
{
    [ColumnName("statusbar")]
    [VectorType(1, 3, ImagePipeline.StatusbarHeight, ImagePipeline.StatusbarWidth)]
    public float[] Statusbar { get; set; } = Array.Empty<float>();
}

internal sealed class DeviceOutput
{
    [ColumnName("probabilities")]
    public float[] Probabilities { get; set; } = Array.Empty<float>();
}

internal sealed record DeviceSetting(int? GpuDeviceId, bool FallbackToCpu, string Requested)
{
    public static DeviceSetting Parse(string value)
    {
        var requested = value.ToLowerInvariant();
        if (requested is "" or "auto")
        {
            return new DeviceSetting(0, true, "auto");
        }
        if (requested == "cpu")
        {
            return new DeviceSetting(null, false, "cpu");
        }
        if (requested == "cuda")
        {
            return new DeviceSetting(0, false, "cuda:0");
        }
        if (requested.StartsWith("cuda:", StringComparison.Ordinal) && int.TryParse(requested[5..], NumberStyles.None, CultureInfo.InvariantCulture, out var id) && id >= 0)
        {
            return new DeviceSetting(id, false, $"cuda:{id}");
        }
        throw new UsageException("--device must be auto, cpu, cuda, or cuda:<non-negative integer>");
    }
}

internal sealed record ModelContract(string FileName, string ModelSha256, string ContractSha256)
{
    public static ModelContract LoadAndVerify(string modelPath, string expectedKind)
    {
        var fullModelPath = Path.GetFullPath(modelPath);
        if (!File.Exists(fullModelPath))
        {
            throw new UsageException($"ONNX model not found: {fullModelPath}");
        }
        var contractPath = Path.ChangeExtension(fullModelPath, ".contract.json");
        if (!File.Exists(contractPath))
        {
            throw new UsageException($"ONNX contract not found: {contractPath}. Deliver the .onnx and .contract.json together.");
        }
        using var document = JsonDocument.Parse(File.ReadAllText(contractPath));
        var root = document.RootElement;
        var kind = root.GetProperty("kind").GetString();
        if (!string.Equals(kind, expectedKind, StringComparison.Ordinal))
        {
            throw new UsageException($"Contract {contractPath} has kind {kind ?? "(missing)"}; expected {expectedKind}");
        }
        var expectedHash = root.GetProperty("onnx").GetProperty("sha256").GetString();
        var actualHash = Sha256(fullModelPath);
        if (string.IsNullOrWhiteSpace(expectedHash) || !string.Equals(expectedHash, actualHash, StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException($"ONNX SHA-256 does not match contract: {fullModelPath}");
        }
        return new ModelContract(
            Path.GetFileName(contractPath),
            actualHash,
            Sha256(contractPath));
    }

    private static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        using var algorithm = SHA256.Create();
        return Convert.ToHexString(algorithm.ComputeHash(stream)).ToLowerInvariant();
    }
}

internal sealed record CliOptions(
    string DetectorPath,
    string? DeviceModelPath,
    string OcrMode,
    string? OcrBundlePath,
    string? OcrModelPath,
    string? InputPath,
    string? InputListPath,
    string OutputDirectory,
    string Device,
    float ScoreThreshold,
    string AnnotationMode,
    string Rectification,
    bool RequireComplete,
    bool ContinueOnError,
    bool SkipExisting,
    int? DetectorIntraOpThreads,
    int? Limit)
{
    public const string Usage = """
Usage:
  dotnet run --project dotnet/ReceiptMlNet.Cli/ReceiptMlNet.Cli.csproj -- \
    --detector <receipt_lrcnn_v1.onnx> \
    [--device-model <statusbar_device_v1.onnx>] \
    [--ocr none|onnx|unified] \
    [--ocr-bundle <paddle-ocr-delivery-directory>] \
    [--ocr-model <receipt_unified_field_reader_v12.onnx>] \
    (--input <image-or-directory> | --input-list <txt>) --output <directory> \
    [--device auto|cpu|cuda:0] [--score-threshold 0.50] [--annotate all|flagged|none] \
    [--rectification none|max-side-1600] \
    [--detector-intra-op-threads <positive-integer>] \
    [--require-complete] [--continue-on-error] [--skip-existing] [--limit 100]

This .NET CLI runs the receipt/device ONNX models and can optionally run a
verified PP-OCR delivery bundle (--ocr onnx) or a v12 unified five-field OCR
reader (--ocr unified). The two OCR modes are mutually exclusive. Unified OCR
requires its adjacent .labels.json and .contract.json sidecars and emits
review-only delivery values until independently human-calibrated. It writes
JSON and, by default, two annotated JPGs. It does not yet include perspective
screen detection. --rectification max-side-1600 applies the Python-compatible
portrait rule (landscape inputs rotate 90 degrees clockwise), then a full-image
cubic warp after EXIF orientation, and limits the longest side to 1600;
perspective photos still require an externally rectified input.
""";

    public static CliOptions Parse(string[] args)
    {
        string? detector = null;
        string? deviceModel = null;
        var ocrMode = "none";
        string? ocrBundle = null;
        string? ocrModel = null;
        string? input = null;
        string? inputList = null;
        string? output = null;
        var device = "auto";
        var scoreThreshold = 0.50f;
        var annotationMode = "all";
        var rectification = ReceiptRectifier.NoneMode;
        var requireComplete = false;
        var continueOnError = false;
        var skipExisting = false;
        int? detectorIntraOpThreads = null;
        int? limit = null;

        for (var index = 0; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--detector": detector = NextValue(args, ref index); break;
                case "--device-model": deviceModel = NextValue(args, ref index); break;
                case "--ocr": ocrMode = ParseOcrMode(NextValue(args, ref index)); break;
                case "--ocr-bundle": ocrBundle = NextValue(args, ref index); break;
                case "--ocr-model": ocrModel = NextValue(args, ref index); break;
                case "--input": input = NextValue(args, ref index); break;
                case "--input-list": inputList = NextValue(args, ref index); break;
                case "--output": output = NextValue(args, ref index); break;
                case "--device": device = NextValue(args, ref index); break;
                case "--annotate": annotationMode = ParseAnnotationMode(NextValue(args, ref index)); break;
                case "--rectification": rectification = ParseRectification(NextValue(args, ref index)); break;
                case "--score-threshold":
                    if (!float.TryParse(NextValue(args, ref index), NumberStyles.Float, CultureInfo.InvariantCulture, out scoreThreshold) || scoreThreshold is < 0.0f or > 1.0f)
                    {
                        throw new UsageException("--score-threshold must be between 0 and 1");
                    }
                    break;
                case "--detector-intra-op-threads":
                    if (!int.TryParse(NextValue(args, ref index), NumberStyles.None, CultureInfo.InvariantCulture, out var parsedDetectorThreads) || parsedDetectorThreads <= 0)
                    {
                        throw new UsageException("--detector-intra-op-threads must be a positive integer");
                    }
                    detectorIntraOpThreads = parsedDetectorThreads;
                    break;
                case "--limit":
                    if (!int.TryParse(NextValue(args, ref index), NumberStyles.None, CultureInfo.InvariantCulture, out var parsedLimit) || parsedLimit <= 0)
                    {
                        throw new UsageException("--limit must be a positive integer");
                    }
                    limit = parsedLimit;
                    break;
                case "--require-complete": requireComplete = true; break;
                case "--continue-on-error": continueOnError = true; break;
                case "--skip-existing": skipExisting = true; break;
                case "--help" or "-h": throw new UsageException(Usage);
                default: throw new UsageException($"Unknown argument: {args[index]}");
            }
        }
        if (string.IsNullOrWhiteSpace(detector) || string.IsNullOrWhiteSpace(output))
        {
            throw new UsageException("--detector and --output are required");
        }
        var hasInput = !string.IsNullOrWhiteSpace(input);
        var hasInputList = !string.IsNullOrWhiteSpace(inputList);
        if (hasInput == hasInputList)
        {
            throw new UsageException("Specify exactly one of --input or --input-list");
        }
        if (ocrMode == "onnx" && string.IsNullOrWhiteSpace(ocrBundle))
        {
            throw new UsageException("--ocr-bundle is required when --ocr onnx");
        }
        if (ocrMode == "unified" && string.IsNullOrWhiteSpace(ocrModel))
        {
            throw new UsageException("--ocr-model is required when --ocr unified");
        }
        if (ocrMode != "onnx" && !string.IsNullOrWhiteSpace(ocrBundle))
        {
            throw new UsageException("--ocr-bundle requires --ocr onnx");
        }
        if (ocrMode != "unified" && !string.IsNullOrWhiteSpace(ocrModel))
        {
            throw new UsageException("--ocr-model requires --ocr unified");
        }
        var parsedDevice = DeviceSetting.Parse(device);
        if (detectorIntraOpThreads is not null && parsedDevice.Requested != "cpu")
        {
            throw new UsageException("--detector-intra-op-threads requires --device cpu");
        }
        return new CliOptions(detector, deviceModel, ocrMode, ocrBundle, ocrModel, input, inputList, output, device, scoreThreshold, annotationMode, rectification, requireComplete, continueOnError, skipExisting, detectorIntraOpThreads, limit);
    }

    private static string ParseOcrMode(string value)
    {
        var mode = value.ToLowerInvariant();
        if (mode is "none" or "onnx" or "unified")
        {
            return mode;
        }
        throw new UsageException("--ocr must be none, onnx, or unified");
    }

    private static string ParseAnnotationMode(string value)
    {
        var mode = value.ToLowerInvariant();
        if (mode is "all" or "flagged" or "none")
        {
            return mode;
        }
        throw new UsageException("--annotate must be all, flagged, or none");
    }

    private static string ParseRectification(string value)
    {
        var mode = value.ToLowerInvariant();
        if (mode is ReceiptRectifier.NoneMode or ReceiptRectifier.MaxSide1600Mode)
        {
            return mode;
        }
        throw new UsageException("--rectification must be none or max-side-1600");
    }

    private static string NextValue(string[] args, ref int index)
    {
        if (++index >= args.Length || args[index].StartsWith("--", StringComparison.Ordinal))
        {
            throw new UsageException($"Missing value for {args[index - 1]}");
        }
        return args[index];
    }
}

internal sealed class UsageException(string message) : Exception(message);

internal sealed record ReceiptResult(
    string Source,
    string InferenceEngine,
    DetectorGeometry Geometry,
    List<DetectionResult> Detections,
    ReceiptFields? Fields,
    DeviceResult? Device,
    ContractReferences? ModelContracts,
    string[] Limitations);

internal sealed record DetectorGeometry(
    ImageSize SourceSize,
    ImageSize RectifiedSize,
    ImageSize DetectorCanvas,
    string ResizeMode,
    string Rectification,
    int RotationDegrees,
    bool ScreenDetected,
    float[][] ScreenQuadOriginal,
    [property: JsonPropertyName("H_original_to_rectified")]
    double[][] HOriginalToRectified,
    [property: JsonPropertyName("H_rectified_to_original")]
    double[][] HRectifiedToOriginal);
internal sealed record ImageSize(int Width, int Height);
internal sealed record DetectionResult(string Label, float Score, float[] BboxImage, OcrResult? Ocr = null);
internal sealed record OcrResult(
    string Text,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] float? Confidence);

internal sealed record ReceiptFields(
    ReceiptFieldResult Time,
    ReceiptFieldResult Amount,
    ReceiptFieldResult TransferStatus,
    ReceiptFieldResult Recipient,
    ReceiptFieldResult PaymentMethod);

/// <summary>JSON-compatible structured field state mirroring the Python pipeline.</summary>
internal sealed record ReceiptFieldResult(
    string State,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] string? Raw,
    float? OcrConfidence,
    float? DetectorScore,
    float? Score,
    string? Value,
    string? Normalized,
    long? AmountFen,
    string? Currency,
    string? Candidate = null,
    string? CtcCandidate = null,
    float? CtcConfidence = null,
    string? StructuredCandidate = null,
    float? StructuredConfidence = null,
    string? DeliveryPolicy = null,
    string? DeliveryValue = null);
internal sealed record DeviceResult(
    string Platform,
    string PlatformCn,
    string Source,
    float Confidence,
    bool DevicePriorConflict,
    float? PIos,
    string? CnnPlatform,
    string? ConflictDetail);
internal sealed record ContractReferences(
    string Detector,
    string? Device,
    string? OcrBundle,
    string? DetectorSha256 = null,
    string? DetectorContractSha256 = null,
    string? DeviceSha256 = null,
    string? DeviceContractSha256 = null,
    string? OcrBundleContractSha256 = null,
    string? UnifiedOcrModel = null,
    string? UnifiedOcrContract = null,
    string? UnifiedOcrModelSha256 = null,
    string? UnifiedOcrLabelsSha256 = null,
    string? UnifiedOcrContractSha256 = null);
internal sealed record ManifestRecord(
    string Source,
    string Result,
    string Status,
    string? AnnotatedRectified = null,
    string? AnnotatedOriginal = null,
    double? InferenceMs = null,
    InferenceStageLatency? StageLatencyMs = null);
internal sealed record ErrorRecord(string Source, string ErrorType, string Message);
internal sealed record InputWorkItem(string Source, string Output);
internal sealed record InferenceSummary(
    string RequestedDevice,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] string? UnifiedProvider,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] int? DetectorIntraOpThreads,
    int Input,
    int Written,
    int Skipped,
    int Errors,
    double TotalSeconds,
    LatencySummary InferenceLatencyMs,
    InferenceStageLatencySummary StageLatencyMs);
internal sealed record LatencySummary(
    int Count,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] double? Mean,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] double? P50,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] double? P95);
internal sealed record InferenceStageLatency(
    double ImageLoad,
    double? Device,
    double DetectorPreprocess,
    double DetectorInference,
    double DetectorPostprocess,
    double? PaddleOcr,
    double? UnifiedOcrPreprocess,
    double? UnifiedOcrInference,
    double? UnifiedOcrPostprocess,
    double ResultAssembly);
internal sealed record InferenceStageLatencySummary(
    LatencySummary ImageLoad,
    LatencySummary Device,
    LatencySummary DetectorPreprocess,
    LatencySummary DetectorInference,
    LatencySummary DetectorPostprocess,
    LatencySummary PaddleOcr,
    LatencySummary UnifiedOcrPreprocess,
    LatencySummary UnifiedOcrInference,
    LatencySummary UnifiedOcrPostprocess,
    LatencySummary ResultAssembly);
