using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
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
        var detectorContract = ModelContract.LoadAndVerify(options.DetectorPath, "receipt_lrcnn_v1");
        ModelContract? deviceContract = options.DeviceModelPath is null
            ? null
            : ModelContract.LoadAndVerify(options.DeviceModelPath, "statusbar_device_v1");
        var ocrBundle = options.OcrMode == "onnx"
            ? PaddleOcrDeliveryBundle.LoadAndVerify(options.OcrBundlePath!)
            : null;

        var inputFiles = EnumerateInputFiles(options.InputPath).ToList();
        if (inputFiles.Count == 0)
        {
            throw new UsageException($"No supported image files found under {options.InputPath}");
        }
        if (options.Limit is not null)
        {
            inputFiles = inputFiles.Take(options.Limit.Value).ToList();
        }

        Directory.CreateDirectory(options.OutputDirectory);
        var sourceRoot = File.Exists(options.InputPath)
            ? Path.GetDirectoryName(Path.GetFullPath(options.InputPath))!
            : Path.GetFullPath(options.InputPath);
        var manifest = new List<ManifestRecord>();
        var errorsPath = Path.Combine(options.OutputDirectory, "inference_errors.jsonl");
        File.WriteAllText(errorsPath, string.Empty, Encoding.UTF8);

        var device = DeviceSetting.Parse(options.Device);
        var detector = new DetectorModel(options.DetectorPath, device);
        var deviceClassifier = options.DeviceModelPath is null
            ? null
            : new DeviceModel(options.DeviceModelPath, device);
        Console.WriteLine($"Requested ONNX device: {device.Requested} (receipt detector{(deviceClassifier is null ? string.Empty : "/device model")})");
        using var ocrEngine = ocrBundle is null ? null : new PaddleOcrEngine(ocrBundle, device);
        if (ocrEngine is not null)
        {
            Console.WriteLine($"OCR ONNX execution provider: {ocrEngine.ExecutionProvider} (det/cls/rec)");
        }

        foreach (var inputFile in inputFiles)
        {
            var outputFile = OutputPathFor(options.OutputDirectory, sourceRoot, inputFile);
            var annotationPaths = AnnotationPaths.ForResultJson(outputFile);
            if (options.SkipExisting && ExistingResultSatisfiesRequestedMode(outputFile, options.OcrMode))
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
                var result = InferImage(inputFile, detector, deviceClassifier, ocrEngine, options.ScoreThreshold);
                if (options.RequireComplete)
                {
                    EnsureCoreFields(result.Detections);
                }
                result = result with
                {
                    ModelContracts = new ContractReferences(
                        detectorContract.FileName,
                        deviceContract?.FileName,
                        ocrBundle is null ? null : Path.GetFileName(ocrBundle.ContractPath)),
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
                    annotationPaths.Original));
            }
            catch (Exception exception)
            {
                var error = new ErrorRecord(Path.GetFullPath(inputFile), exception.GetType().Name, exception.Message);
                File.AppendAllText(errorsPath, JsonSerializer.Serialize(error, JsonOptions) + Environment.NewLine, Encoding.UTF8);
                if (!options.ContinueOnError)
                {
                    throw;
                }
            }
        }

        WriteJsonAtomic(Path.Combine(options.OutputDirectory, "inference_manifest.json"), manifest);
        Console.WriteLine($"Wrote {manifest.Count(record => record.Status == "written")} ML.NET result bundle(s) to {options.OutputDirectory}");
    }

    private static ReceiptResult InferImage(
        string inputFile,
        DetectorModel detector,
        DeviceModel? deviceClassifier,
        PaddleOcrEngine? ocrEngine,
        float scoreThreshold)
    {
        using var source = ImagePipeline.LoadUprightRgb(inputFile);
        var sourceSize = new ImageSize(source.Width, source.Height);
        var device = deviceClassifier?.Classify(source);
        var prepared = ImagePipeline.PrepareDetectorInput(source);
        var predictions = detector.Predict(prepared.Tensor);
        var detections = PostProcessDetections(predictions, prepared, scoreThreshold);
        if (ocrEngine is not null)
        {
            detections = EnrichWithOcr(source, detections, ocrEngine);
        }
        var fields = ocrEngine is null ? null : BuildFields(detections);

        return new ReceiptResult(
            Path.GetFullPath(inputFile),
            "mlnet",
            new DetectorGeometry(
                sourceSize,
                new ImageSize(ImagePipeline.DetectorWidth, ImagePipeline.DetectorHeight),
                "letterbox",
                "not_applied"),
            detections,
            fields,
            device,
            null,
            new[]
            {
                "This .NET CLI performs ONNX model inference.",
                "Input must already be an upright, rectified receipt image when perspective correction is needed.",
                "Annotated JPGs use upright source coordinates. Their original/rectified pair is identical until perspective rectification is ported to .NET.",
                ocrEngine is null
                    ? "PaddleOCR ONNX field extraction is disabled; pass --ocr onnx --ocr-bundle <delivery-directory> to enable it."
                    : "OCR uses the verified PP-OCR ONNX delivery bundle; source-image perspective rectification is not yet ported to .NET.",
            });
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
            var crop = CropFieldWithMargin(source, detection.BboxImage);
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

    private static Image<Rgb24>? CropFieldWithMargin(Image<Rgb24> source, float[] box)
    {
        if (box.Length < 4)
        {
            return null;
        }
        var marginX = Math.Max(2.0f, (box[2] - box[0]) * 0.08f);
        var marginY = Math.Max(2.0f, (box[3] - box[1]) * 0.08f);
        var left = Math.Clamp((int)MathF.Floor(box[0] - marginX), 0, source.Width);
        var top = Math.Clamp((int)MathF.Floor(box[1] - marginY), 0, source.Height);
        var right = Math.Clamp((int)MathF.Ceiling(box[2] + marginX), 0, source.Width);
        var bottom = Math.Clamp((int)MathF.Ceiling(box[3] + marginY), 0, source.Height);
        if (right <= left || bottom <= top)
        {
            return null;
        }
        return source.Clone(context => context.Crop(new Rectangle(left, top, right - left, bottom - top)));
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

    private static bool ExistingResultSatisfiesRequestedMode(string outputPath, string ocrMode)
    {
        if (!File.Exists(outputPath) || ocrMode == "none")
        {
            return File.Exists(outputPath);
        }
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(outputPath));
            return document.RootElement.TryGetProperty("fields", out var fields)
                && fields.ValueKind == JsonValueKind.Object;
        }
        catch (JsonException)
        {
            return false;
        }
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

    private static string OutputPathFor(string outputDirectory, string sourceRoot, string sourcePath)
    {
        var relative = Path.GetRelativePath(sourceRoot, sourcePath);
        if (relative.StartsWith("..", StringComparison.Ordinal))
        {
            relative = Path.GetFileName(sourcePath);
        }
        return Path.Combine(outputDirectory, Path.ChangeExtension(relative, ".json"));
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
        "amount", "transfer_status", "recipient_field", "payment_method_field",
    };

    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    };
}

internal sealed class DetectorModel
{
    private readonly PredictionEngine<DetectorInput, DetectorOutput> _engine;

    public DetectorModel(string modelPath, DeviceSetting device)
    {
        var context = new MLContext(seed: 1);
        var fitData = context.Data.LoadFromEnumerable(new[]
        {
            new DetectorInput { Image = new float[3 * ImagePipeline.DetectorHeight * ImagePipeline.DetectorWidth] },
        });
        var pipeline = context.Transforms.ApplyOnnxModel(
            outputColumnNames: new[] { "boxes", "labels", "scores" },
            inputColumnNames: new[] { "image" },
            modelFile: modelPath,
            gpuDeviceId: device.GpuDeviceId,
            fallbackToCpu: device.FallbackToCpu);
        var model = pipeline.Fit(fitData);
        _engine = context.Model.CreatePredictionEngine<DetectorInput, DetectorOutput>(model);
    }

    public DetectorOutput Predict(float[] tensor) => _engine.Predict(new DetectorInput { Image = tensor });

}

internal sealed class DeviceModel
{
    private readonly PredictionEngine<DeviceInput, DeviceOutput> _engine;

    public DeviceModel(string modelPath, DeviceSetting device)
    {
        var context = new MLContext(seed: 1);
        var fitData = context.Data.LoadFromEnumerable(new[]
        {
            new DeviceInput { Statusbar = new float[1 * 3 * ImagePipeline.StatusbarHeight * ImagePipeline.StatusbarWidth] },
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
        var probabilities = _engine.Predict(new DeviceInput { Statusbar = ImagePipeline.PrepareStatusbarInput(source) }).Probabilities;
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
    public const int StatusbarWidth = 512;
    public const int StatusbarHeight = 64;

    public static Image<Rgb24> LoadUprightRgb(string path)
    {
        var image = Image.Load<Rgb24>(path);
        image.Mutate(context => context.AutoOrient());
        return image;
    }

    public static DetectorInputTensor PrepareDetectorInput(Image<Rgb24> source)
    {
        var scale = Math.Min((float)DetectorWidth / source.Width, (float)DetectorHeight / source.Height);
        var resizedWidth = Math.Clamp((int)Math.Round(source.Width * scale, MidpointRounding.ToEven), 1, DetectorWidth);
        var resizedHeight = Math.Clamp((int)Math.Round(source.Height * scale, MidpointRounding.ToEven), 1, DetectorHeight);
        var left = (DetectorWidth - resizedWidth) / 2;
        var top = (DetectorHeight - resizedHeight) / 2;
        // ImageSharp calls the bilinear kernel "Triangle".
        using var resized = source.Clone(context => context.Resize(resizedWidth, resizedHeight, KnownResamplers.Triangle));
        using var canvas = new Image<Rgb24>(DetectorWidth, DetectorHeight);
        canvas.Mutate(context => context.DrawImage(resized, new Point(left, top), 1.0f));

        var values = new float[3 * DetectorHeight * DetectorWidth];
        var plane = DetectorHeight * DetectorWidth;
        for (var y = 0; y < DetectorHeight; y++)
        {
            for (var x = 0; x < DetectorWidth; x++)
            {
                var pixel = canvas[x, y];
                var offset = y * DetectorWidth + x;
                values[offset] = pixel.R / 255.0f;
                values[plane + offset] = pixel.G / 255.0f;
                values[2 * plane + offset] = pixel.B / 255.0f;
            }
        }
        return new DetectorInputTensor(values, source.Width, source.Height, (float)resizedWidth / source.Width, (float)resizedHeight / source.Height, left, top);
    }

    public static float[] PrepareStatusbarInput(Image<Rgb24> source)
    {
        var stripHeight = Math.Max(1, (int)Math.Round(source.Height * 0.08, MidpointRounding.ToEven));
        using var strip = source.Clone(context => context.Crop(new Rectangle(0, 0, source.Width, stripHeight)));
        // The Python training/inference path calls Pillow resize without an
        // explicit filter for RGB, whose default is bicubic.
        using var canvas = strip.Clone(context => context.Resize(StatusbarWidth, StatusbarHeight, KnownResamplers.Bicubic));
        var values = new float[3 * StatusbarHeight * StatusbarWidth];
        var plane = StatusbarHeight * StatusbarWidth;
        for (var y = 0; y < StatusbarHeight; y++)
        {
            for (var x = 0; x < StatusbarWidth; x++)
            {
                var pixel = canvas[x, y];
                var offset = y * StatusbarWidth + x;
                values[offset] = (pixel.R / 255.0f - 0.485f) / 0.229f;
                values[plane + offset] = (pixel.G / 255.0f - 0.456f) / 0.224f;
                values[2 * plane + offset] = (pixel.B / 255.0f - 0.406f) / 0.225f;
            }
        }
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

internal sealed class DetectorInput
{
    [ColumnName("image")]
    [VectorType(3, ImagePipeline.DetectorHeight, ImagePipeline.DetectorWidth)]
    public float[] Image { get; set; } = Array.Empty<float>();
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

internal sealed record ModelContract(string FileName)
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
        return new ModelContract(Path.GetFileName(contractPath));
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
    string InputPath,
    string OutputDirectory,
    string Device,
    float ScoreThreshold,
    string AnnotationMode,
    bool RequireComplete,
    bool ContinueOnError,
    bool SkipExisting,
    int? Limit)
{
    public const string Usage = """
Usage:
  dotnet run --project dotnet/ReceiptMlNet.Cli/ReceiptMlNet.Cli.csproj -- \
    --detector <receipt_lrcnn_v1.onnx> \
    [--device-model <statusbar_device_v1.onnx>] \
    [--ocr none|onnx] [--ocr-bundle <paddle-ocr-delivery-directory>] \
    --input <image-or-directory> --output <directory> \
    [--device auto|cpu|cuda:0] [--score-threshold 0.50] [--annotate all|flagged|none] \
    [--require-complete] [--continue-on-error] [--skip-existing] [--limit 100]

This .NET CLI runs the receipt/device ONNX models and can optionally run a
verified PP-OCR ONNX delivery bundle. It writes JSON and, by default, two
annotated JPGs. It does not yet include perspective rectification; use an
already rectified image when needed.
""";

    public static CliOptions Parse(string[] args)
    {
        string? detector = null;
        string? deviceModel = null;
        var ocrMode = "none";
        string? ocrBundle = null;
        string? input = null;
        string? output = null;
        var device = "auto";
        var scoreThreshold = 0.50f;
        var annotationMode = "all";
        var requireComplete = false;
        var continueOnError = false;
        var skipExisting = false;
        int? limit = null;

        for (var index = 0; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--detector": detector = NextValue(args, ref index); break;
                case "--device-model": deviceModel = NextValue(args, ref index); break;
                case "--ocr": ocrMode = ParseOcrMode(NextValue(args, ref index)); break;
                case "--ocr-bundle": ocrBundle = NextValue(args, ref index); break;
                case "--input": input = NextValue(args, ref index); break;
                case "--output": output = NextValue(args, ref index); break;
                case "--device": device = NextValue(args, ref index); break;
                case "--annotate": annotationMode = ParseAnnotationMode(NextValue(args, ref index)); break;
                case "--score-threshold":
                    if (!float.TryParse(NextValue(args, ref index), NumberStyles.Float, CultureInfo.InvariantCulture, out scoreThreshold) || scoreThreshold is < 0.0f or > 1.0f)
                    {
                        throw new UsageException("--score-threshold must be between 0 and 1");
                    }
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
        if (string.IsNullOrWhiteSpace(detector) || string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
        {
            throw new UsageException("--detector, --input and --output are required");
        }
        if (ocrMode == "onnx" && string.IsNullOrWhiteSpace(ocrBundle))
        {
            throw new UsageException("--ocr-bundle is required when --ocr onnx");
        }
        if (ocrMode == "none" && !string.IsNullOrWhiteSpace(ocrBundle))
        {
            throw new UsageException("--ocr-bundle requires --ocr onnx");
        }
        _ = DeviceSetting.Parse(device);
        return new CliOptions(detector, deviceModel, ocrMode, ocrBundle, input, output, device, scoreThreshold, annotationMode, requireComplete, continueOnError, skipExisting, limit);
    }

    private static string ParseOcrMode(string value)
    {
        var mode = value.ToLowerInvariant();
        if (mode is "none" or "onnx")
        {
            return mode;
        }
        throw new UsageException("--ocr must be none or onnx");
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

internal sealed record DetectorGeometry(ImageSize SourceSize, ImageSize DetectorCanvas, string ResizeMode, string Rectification);
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
    string? Currency);
internal sealed record DeviceResult(
    string Platform,
    string PlatformCn,
    string Source,
    float Confidence,
    bool DevicePriorConflict,
    float? PIos,
    string? CnnPlatform,
    string? ConflictDetail);
internal sealed record ContractReferences(string Detector, string? Device, string? OcrBundle);
internal sealed record ManifestRecord(
    string Source,
    string Result,
    string Status,
    string? AnnotatedRectified = null,
    string? AnnotatedOriginal = null);
internal sealed record ErrorRecord(string Source, string ErrorType, string Message);
