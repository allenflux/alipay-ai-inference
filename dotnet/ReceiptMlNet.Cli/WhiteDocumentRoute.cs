using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

internal static class DocumentRoutePolicy
{
    public const string Auto = "auto";
    public const string Blue = "blue";
    public const string White = "white";

    public static string Parse(string value)
    {
        var mode = value.ToLowerInvariant();
        if (mode is Auto or Blue or White)
        {
            return mode;
        }
        throw new UsageException("--document-type must be auto, blue, or white");
    }

    public static void RequireRunnable(string documentType)
    {
        if (documentType == Auto)
        {
            throw new UsageException(
                "--document-type auto is fail-closed: no calibrated blue/white router is delivered; "
                + "choose --document-type blue or --document-type white explicitly");
        }
        if (documentType is not (Blue or White))
        {
            throw new UsageException($"Unsupported document type: {documentType}");
        }
    }
}

/// <summary>
/// The exact status-bar model and contract identity used by the white CPU
/// route. Both files are read once; the parsed contract hash and ONNX session
/// are therefore bound to these private bytes rather than a reopenable path.
/// </summary>
internal sealed class DeviceModelCpuSnapshot
{
    private DeviceModelCpuSnapshot(ModelContract contract, byte[] modelBytes)
    {
        Contract = contract;
        _modelBytes = modelBytes;
    }

    private readonly byte[] _modelBytes;
    internal ModelContract Contract { get; }
    internal string ClosedModelSha256 => Sha256(_modelBytes);

    public static DeviceModelCpuSnapshot LoadAndVerify(string modelPath)
    {
        var fullModelPath = Path.GetFullPath(modelPath);
        if (!File.Exists(fullModelPath))
        {
            throw new UsageException($"ONNX model not found: {fullModelPath}");
        }
        var contractPath = Path.ChangeExtension(fullModelPath, ".contract.json");
        if (!File.Exists(contractPath))
        {
            throw new UsageException(
                $"ONNX contract not found: {contractPath}. Deliver the .onnx and .contract.json together.");
        }

        // Do not hash paths and reopen them for consumption. These are the
        // same bytes parsed below and handed to ONNX Runtime.
        var contractBytes = File.ReadAllBytes(contractPath);
        var modelBytes = File.ReadAllBytes(fullModelPath);
        try
        {
            using var document = JsonDocument.Parse(contractBytes);
            var root = document.RootElement;
            var kind = root.GetProperty("kind").GetString();
            if (!string.Equals(kind, "statusbar_device_v1", StringComparison.Ordinal))
            {
                throw new UsageException(
                    $"Contract {contractPath} has kind {kind ?? "(missing)"}; expected statusbar_device_v1");
            }
            var expectedHash = root.GetProperty("onnx").GetProperty("sha256").GetString();
            if (string.IsNullOrWhiteSpace(expectedHash))
            {
                throw new UsageException($"ONNX SHA-256 is missing from contract: {contractPath}");
            }
            var closedModelBytes = VerifyAndCloneModelBytes(modelBytes, expectedHash);
            var actualHash = Sha256(closedModelBytes);
            return new DeviceModelCpuSnapshot(
                new ModelContract(
                    Path.GetFileName(contractPath),
                    actualHash,
                    Sha256(contractBytes)),
                closedModelBytes);
        }
        catch (JsonException exception)
        {
            throw new UsageException($"Invalid ONNX contract {contractPath}: {exception.Message}");
        }
    }

    internal InferenceSession OpenSession()
    {
        return new InferenceSession(_modelBytes);
    }

    internal static byte[] VerifyAndCloneModelBytes(byte[] bytes, string expectedSha256)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (string.IsNullOrWhiteSpace(expectedSha256)
            || expectedSha256.Length != 64
            || expectedSha256.Any(character => !Uri.IsHexDigit(character)))
        {
            throw new InvalidOperationException("Status-bar model snapshot expected SHA-256 is invalid");
        }
        var actualHash = Sha256(bytes);
        if (!string.Equals(actualHash, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "Status-bar model bytes changed before the private CPU snapshot");
        }
        return bytes.ToArray();
    }

    private static string Sha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}

/// <summary>CPU-only status-bar classifier backed by a closed ONNX byte snapshot.</summary>
internal sealed class ClosedCpuDeviceModel : IDisposable
{
    private static readonly string[] InputNames = ["statusbar"];
    private static readonly string[] OutputNames = ["probabilities"];
    private static readonly long[] InputShape =
        [1, 3, ImagePipeline.StatusbarHeight, ImagePipeline.StatusbarWidth];

    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;
    private readonly float[] _inputBuffer = new float[ImagePipeline.StatusbarTensorLength];
    private readonly OrtValue _inputValue;
    private readonly OrtValue[] _inputValues;
    private bool _disposed;

    public ClosedCpuDeviceModel(DeviceModelCpuSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        var session = snapshot.OpenSession();
        OrtValue? inputValue = null;
        RunOptions? runOptions = null;
        try
        {
            VerifyAbi(session);
            inputValue = OrtValue.CreateTensorValueFromMemory(_inputBuffer, InputShape);
            runOptions = new RunOptions();
            var inputValues = new OrtValue[] { inputValue };
            _session = session;
            _inputValue = inputValue;
            _inputValues = inputValues;
            _runOptions = runOptions;
            inputValue = null;
            runOptions = null;
        }
        catch
        {
            runOptions?.Dispose();
            inputValue?.Dispose();
            session.Dispose();
            throw;
        }
    }

    public DeviceResult Classify(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> source)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ClosedCpuDeviceModel));
        }
        return DeviceModel.ClassifyCore(source, PredictPIos);
    }

    private float PredictPIos(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> source)
    {
        ImagePipeline.PrepareStatusbarInput(source, _inputBuffer);
        using var outputs = _session.Run(_runOptions, InputNames, _inputValues, OutputNames);
        if (outputs.Count != 1)
        {
            throw new InvalidOperationException(
                $"ONNX status-bar model returned {outputs.Count} outputs; expected 1");
        }
        var output = outputs.Single();
        var typeAndShape = output.GetTensorTypeAndShape();
        var probabilities = output.GetTensorDataAsSpan<float>();
        if (typeAndShape.ElementDataType != TensorElementType.Float
            || typeAndShape.ElementCount != 2
            || probabilities.Length != 2)
        {
            throw new InvalidOperationException(
                "ONNX status-bar model probabilities must contain exactly two floats");
        }
        return probabilities[1];
    }

    private static void VerifyAbi(InferenceSession session)
    {
        if (!session.InputNames.SequenceEqual(InputNames, StringComparer.Ordinal)
            || !session.OutputNames.SequenceEqual(OutputNames, StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                "ONNX status-bar model must expose exactly statusbar -> probabilities");
        }
        var input = session.InputMetadata[InputNames[0]];
        var expected = new[] { 1, 3, ImagePipeline.StatusbarHeight, ImagePipeline.StatusbarWidth };
        if (!input.IsTensor
            || input.ElementType != typeof(float)
            || input.Dimensions.Length != expected.Length
            || input.Dimensions.Where((dimension, index) => dimension != -1 && dimension != expected[index]).Any())
        {
            throw new InvalidOperationException(
                "ONNX status-bar model input must be float [1,3,64,512] with optional dynamic batch");
        }
        var output = session.OutputMetadata[OutputNames[0]];
        if (!output.IsTensor
            || output.ElementType != typeof(float)
            || output.Dimensions.Length == 0
            || (output.Dimensions[^1] != -1 && output.Dimensions[^1] != 2))
        {
            throw new InvalidOperationException(
                "ONNX status-bar model output probabilities must end in two floats");
        }
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
}

internal static class WhiteDocumentOutputContract
{
    public const int SchemaVersion = 1;
    public const string SemanticsVersion = "white_document_paddle_student_layout_review_v2";
    public const string DeliveryPolicy = "review_only";
    public const string StudentCropSource = "same_paddle_db_cls_oriented_crop";

    public static IReadOnlyList<WhiteDocumentLine> ProjectLines(PaddleOcrLayoutReadResult read)
    {
        ArgumentNullException.ThrowIfNull(read);
        return read.Lines
            .Select((line, index) => new WhiteDocumentLine(
                index,
                line.Text,
                MathF.Round(line.Confidence, 6),
                line.PassesDropScore,
                line.Quad
                    .Select(point => new WhiteDocumentPoint(
                        MathF.Round(point.X, 4),
                        MathF.Round(point.Y, 4)))
                    .ToArray(),
                line.Student is null
                    ? null
                    : new WhiteDocumentStudentLineEvidence(
                        line.Student.Text,
                        MathF.Round(line.Student.Confidence, 6),
                        NormalizedExactMatch(line.Text, line.Student.Text),
                        "cpu",
                        DeliveryPolicy,
                        StudentCropSource)))
            .ToArray();
    }

    internal static bool NormalizedExactMatch(string teacher, string student) =>
        string.Equals(NormalizeComparison(teacher), NormalizeComparison(student), StringComparison.Ordinal);

    private static string NormalizeComparison(string value)
    {
        var normalized = (value ?? string.Empty).Normalize(NormalizationForm.FormKC);
        var output = new StringBuilder(normalized.Length);
        var pendingSpace = false;
        foreach (var character in normalized)
        {
            if (char.IsWhiteSpace(character))
            {
                pendingSpace = output.Length > 0;
                continue;
            }
            if (pendingSpace)
            {
                output.Append(' ');
                pendingSpace = false;
            }
            output.Append(character);
        }
        return output.ToString();
    }
}

internal static partial class ReceiptMlNetProgram
{
    internal static void RequireFreshWhiteOutput(string outputDirectory)
    {
        if (Directory.Exists(outputDirectory) || File.Exists(outputDirectory))
        {
            throw new UsageException(
                $"White document output already exists: {outputDirectory}. " +
                "Use a brand-new output directory so review evidence cannot be overwritten.");
        }
    }

    private static void RunWhiteDocumentRoute(CliOptions options)
    {
        var totalStopwatch = Stopwatch.StartNew();
        var deviceSnapshot = DeviceModelCpuSnapshot.LoadAndVerify(options.DeviceModelPath!);
        var deviceContract = deviceSnapshot.Contract;
        var ocrBundle = PaddleOcrDeliveryBundle.LoadAndVerify(options.OcrBundlePath!);
        var ocrSnapshot = PaddleOcrCpuRuntimeSnapshot.LoadAndVerify(ocrBundle);
        var studentBundle = string.IsNullOrWhiteSpace(options.WhiteStudentBundlePath)
            ? null
            : WhiteLineStudentBundle.LoadAndVerify(options.WhiteStudentBundlePath);
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

        RequireFreshWhiteOutput(options.OutputDirectory);
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

        var device = DeviceSetting.Parse(options.Device);
        var manifest = new List<WhiteDocumentManifestRecord>();
        var inferenceLatencies = new List<double>();
        var stageLatencies = new List<WhiteDocumentStageLatency>();
        var errorCount = 0;
        Exception? fatalException = null;
        var errorsPath = Path.Combine(options.OutputDirectory, "inference_errors.jsonl");
        File.WriteAllText(errorsPath, string.Empty, Encoding.UTF8);

        using var deviceClassifier = new ClosedCpuDeviceModel(deviceSnapshot);
        using var ocrEngine = new PaddleOcrEngine(ocrBundle, ocrSnapshot.Models);
        using var studentEngine = studentBundle is null
            ? null
            : new WhiteLineStudentEngine(studentBundle);
        if (!string.Equals(ocrEngine.ExecutionProvider, "cpu", StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"White document route requires CPU PP-OCR; provider was {ocrEngine.ExecutionProvider}");
        }
        Console.WriteLine("Document route: explicit white (review-only full-image line OCR; no five-field mapping)");
        Console.WriteLine("Status-bar device classification runs before white document OCR");
        Console.WriteLine($"OCR ONNX execution provider: {ocrEngine.ExecutionProvider} (det/cls/rec)");
        Console.WriteLine(studentEngine is null
            ? "White line student: not configured (Paddle review evidence only)"
            : "White line student: CPU comparison enabled on the same Paddle DB/CLS-oriented crops (review-only)");
        if (!string.IsNullOrWhiteSpace(options.DetectorPath))
        {
            Console.WriteLine("The supplied blue receipt detector is intentionally not loaded by the explicit white route");
        }

        foreach (var workItem in workItems)
        {
            if (options.SkipExisting && ExistingWhiteDocumentResultSatisfiesRequestedMode(
                    workItem.Output,
                    deviceContract,
                    ocrBundle,
                    studentBundle))
            {
                manifest.Add(new WhiteDocumentManifestRecord(
                    Path.GetFullPath(workItem.Source),
                    workItem.Output,
                    "skipped_existing"));
                continue;
            }

            try
            {
                var inferenceStopwatch = Stopwatch.StartNew();
                var result = InferWhiteDocument(
                    workItem.Source,
                    deviceClassifier,
                    ocrEngine,
                    studentEngine,
                    deviceContract,
                    ocrBundle,
                    ocrSnapshot,
                    studentBundle,
                    out var stageLatency);
                inferenceStopwatch.Stop();
                var inferenceMs = Math.Round(inferenceStopwatch.Elapsed.TotalMilliseconds, 4);
                WriteJsonAtomic(workItem.Output, result);
                manifest.Add(new WhiteDocumentManifestRecord(
                    Path.GetFullPath(workItem.Source),
                    workItem.Output,
                    "written",
                    inferenceMs,
                    stageLatency));
                inferenceLatencies.Add(inferenceMs);
                stageLatencies.Add(stageLatency);
            }
            catch (Exception exception)
            {
                errorCount++;
                var error = new ErrorRecord(
                    Path.GetFullPath(workItem.Source),
                    exception.GetType().Name,
                    exception.Message);
                File.AppendAllText(
                    errorsPath,
                    JsonSerializer.Serialize(error, JsonOptions) + Environment.NewLine,
                    Encoding.UTF8);
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
        var summary = new WhiteDocumentInferenceSummary(
            DocumentRoutePolicy.White,
            device.Requested,
            ocrEngine.ExecutionProvider,
            workItems.Count,
            writtenCount,
            skippedCount,
            errorCount,
            Math.Round(totalStopwatch.Elapsed.TotalSeconds, 4),
            SummarizeLatencies(inferenceLatencies),
            new WhiteDocumentStageLatencySummary(
                SummarizeLatencies(stageLatencies.Select(item => item.ImageLoad).ToArray()),
                SummarizeLatencies(stageLatencies.Select(item => item.Device).ToArray()),
                SummarizeLatencies(stageLatencies.Select(item => item.PaddleOcr).ToArray()),
                SummarizeLatencies(stageLatencies.Select(item => item.ResultAssembly).ToArray())),
            studentEngine?.ExecutionProvider);
        WriteJsonAtomic(Path.Combine(options.OutputDirectory, "inference_summary.json"), summary);
        if (fatalException is not null)
        {
            ExceptionDispatchInfo.Capture(fatalException).Throw();
        }
        Console.WriteLine($"Wrote {writtenCount} review-only white document result bundle(s) to {options.OutputDirectory}");
    }

    private static WhiteDocumentResult InferWhiteDocument(
        string inputFile,
        ClosedCpuDeviceModel deviceClassifier,
        PaddleOcrEngine ocrEngine,
        WhiteLineStudentEngine? studentEngine,
        ModelContract deviceContract,
        PaddleOcrDeliveryBundle ocrBundle,
        PaddleOcrCpuRuntimeSnapshot ocrSnapshot,
        WhiteLineStudentBundle? studentBundle,
        out WhiteDocumentStageLatency stageLatency)
    {
        var stageStopwatch = Stopwatch.StartNew();
        using var source = ImagePipeline.LoadUprightRgb(inputFile);
        var imageLoadMs = StopAndReadMilliseconds(stageStopwatch);

        // Device classification intentionally precedes every body/layout step
        // so the same status-bar model is shared by blue and white documents.
        stageStopwatch.Restart();
        var device = deviceClassifier.Classify(source);
        var deviceMs = StopAndReadMilliseconds(stageStopwatch);

        stageStopwatch.Restart();
        var layout = studentEngine is null
            ? ocrEngine.RecognizeLayoutDiagnostic(source)
            : ocrEngine.RecognizeLayoutWithStudentDiagnostic(source, studentEngine);
        var paddleOcrMs = StopAndReadMilliseconds(stageStopwatch);

        stageStopwatch.Restart();
        var lines = WhiteDocumentOutputContract.ProjectLines(layout);
        var resultAssemblyMs = StopAndReadMilliseconds(stageStopwatch);
        stageLatency = new WhiteDocumentStageLatency(
            imageLoadMs,
            deviceMs,
            paddleOcrMs,
            resultAssemblyMs);

        return new WhiteDocumentResult(
            WhiteDocumentOutputContract.SchemaVersion,
            WhiteDocumentOutputContract.SemanticsVersion,
            DocumentRoutePolicy.White,
            Path.GetFullPath(inputFile),
            "dotnet_onnxruntime_cpu",
            new ImageSize(source.Width, source.Height),
            new WhiteDocumentRouteEvidence(
                DocumentRoutePolicy.White,
                DocumentRoutePolicy.White,
                "explicit_cli",
                false,
                true,
                "white_field_mapping_not_calibrated"),
            new WhiteDocumentOcrEvidence(
                "paddle_teacher_candidate",
                "ppocr_db_cls_rec",
                ocrEngine.ExecutionProvider,
                WhiteDocumentOutputContract.DeliveryPolicy,
                studentEngine is null ? "not_configured" : "integrated_review_only",
                "not_calibrated",
                layout.Text,
                layout.Confidence is null ? null : MathF.Round(layout.Confidence.Value, 6),
                lines.Count(line => line.PassesDropScore),
                lines.Count,
                studentEngine?.ExecutionProvider,
                lines.Count(line => line.Student is not null),
                lines.Count(line => line.Student?.NormalizedExactMatch == true),
                studentEngine is null ? null : WhiteDocumentOutputContract.StudentCropSource),
            lines,
            device,
            new WhiteDocumentContractReferences(
                deviceContract.FileName,
                deviceContract.ModelSha256,
                deviceContract.ContractSha256,
                Path.GetFileName(ocrBundle.ContractPath),
                ocrBundle.ContractSha256,
                ocrBundle.SourceAuditContractSha256,
                ocrBundle.DetModel.File.RelativePath,
                ocrBundle.DetModel.File.Sha256,
                ocrBundle.ClsModel.File.RelativePath,
                ocrBundle.ClsModel.File.Sha256,
                ocrBundle.RecModel.File.RelativePath,
                ocrBundle.RecModel.File.Sha256,
                ocrBundle.Dictionary.RelativePath,
                ocrBundle.Dictionary.Sha256,
                "immutable_verified_bytes",
                false,
                ocrSnapshot.DictionarySizeBytes,
                ocrSnapshot.DictionarySha256,
                studentBundle?.ModelFileName,
                studentBundle?.ModelSha256,
                studentBundle?.ModelSizeBytes,
                studentBundle?.CharsetFileName,
                studentBundle?.CharsetSha256,
                studentBundle?.CharsetSizeBytes,
                studentBundle?.ContractFileName,
                studentBundle?.ContractSha256,
                studentBundle?.ContractSizeBytes,
                studentBundle is null ? null : "immutable_verified_bytes",
                studentBundle is null ? null : false),
            stageLatency,
            [
                "EXIF orientation is applied before status-bar classification and full-image OCR.",
                "White output is line-level Paddle teacher evidence with an optional CPU student comparison; it does not fabricate the blue receipt five-field schema.",
                "The optional student receives the exact DB crop after Paddle CLS orientation used by Paddle REC; it does not run a second crop or angle pipeline.",
                "All Paddle and student OCR lines remain review-only, including lines above the delivery bundle drop score.",
                "Status-bar and PP-OCR sessions consume hash-verified immutable byte snapshots; model and dictionary paths are not reopened after closure.",
                "Teacher agreement is not independent human ground truth.",
                "Automatic blue/white routing remains unavailable until a separately calibrated router is delivered.",
            ]);
    }

    private static bool ExistingWhiteDocumentResultSatisfiesRequestedMode(
        string outputPath,
        ModelContract deviceContract,
        PaddleOcrDeliveryBundle ocrBundle,
        WhiteLineStudentBundle? studentBundle)
    {
        if (!File.Exists(outputPath))
        {
            return false;
        }
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(outputPath));
            var root = document.RootElement;
            if (!root.TryGetProperty("result_schema_version", out var schema)
                || !schema.TryGetInt32(out var schemaVersion)
                || schemaVersion != WhiteDocumentOutputContract.SchemaVersion
                || !HasJsonString(root, "result_semantics_version", WhiteDocumentOutputContract.SemanticsVersion)
                || !HasJsonString(root, "document_type", DocumentRoutePolicy.White)
                || root.TryGetProperty("fields", out _)
                || root.TryGetProperty("detections", out _)
                || !root.TryGetProperty("route", out var route)
                || !HasJsonString(route, "resolved_document_type", DocumentRoutePolicy.White)
                || !HasJsonBool(route, "review_required", true)
                || !root.TryGetProperty("ocr", out var ocr)
                || !HasJsonString(ocr, "provider", "cpu")
                || !HasJsonString(ocr, "delivery_policy", WhiteDocumentOutputContract.DeliveryPolicy)
                || !HasJsonString(
                    ocr,
                    "student_model_status",
                    studentBundle is null ? "not_configured" : "integrated_review_only")
                || !root.TryGetProperty("lines", out var lines)
                || lines.ValueKind != JsonValueKind.Array
                || !root.TryGetProperty("model_contracts", out var contracts))
            {
                return false;
            }
            return HasJsonString(contracts, "device", deviceContract.FileName)
                && HasJsonString(contracts, "device_sha256", deviceContract.ModelSha256)
                && HasJsonString(contracts, "device_contract_sha256", deviceContract.ContractSha256)
                && HasJsonString(contracts, "ocr_bundle", Path.GetFileName(ocrBundle.ContractPath))
                && HasJsonString(contracts, "ocr_bundle_contract_sha256", ocrBundle.ContractSha256)
                && HasJsonString(contracts, "ocr_detector_sha256", ocrBundle.DetModel.File.Sha256)
                && HasJsonString(contracts, "ocr_classifier_sha256", ocrBundle.ClsModel.File.Sha256)
                && HasJsonString(contracts, "ocr_recognizer_sha256", ocrBundle.RecModel.File.Sha256)
                && HasJsonString(contracts, "ocr_dictionary_sha256", ocrBundle.Dictionary.Sha256)
                && HasJsonString(contracts, "runtime_source", "immutable_verified_bytes")
                && HasJsonBool(contracts, "reopened_paths_after_verification", false)
                && HasJsonLong(
                    contracts,
                    "ocr_dictionary_snapshot_size_bytes",
                    ocrBundle.Dictionary.SizeBytes)
                && HasJsonString(
                    contracts,
                    "ocr_dictionary_snapshot_sha256",
                    ocrBundle.Dictionary.Sha256)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_model",
                    studentBundle?.ModelFileName)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_model_sha256",
                    studentBundle?.ModelSha256)
                && HasOptionalJsonLong(
                    contracts,
                    "white_student_model_snapshot_size_bytes",
                    studentBundle?.ModelSizeBytes)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_charset",
                    studentBundle?.CharsetFileName)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_charset_sha256",
                    studentBundle?.CharsetSha256)
                && HasOptionalJsonLong(
                    contracts,
                    "white_student_charset_snapshot_size_bytes",
                    studentBundle?.CharsetSizeBytes)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_contract",
                    studentBundle?.ContractFileName)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_contract_sha256",
                    studentBundle?.ContractSha256)
                && HasOptionalJsonLong(
                    contracts,
                    "white_student_contract_snapshot_size_bytes",
                    studentBundle?.ContractSizeBytes)
                && HasOptionalJsonString(
                    contracts,
                    "white_student_runtime_source",
                    studentBundle is null ? null : "immutable_verified_bytes")
                && HasOptionalJsonBool(
                    contracts,
                    "white_student_reopened_paths_after_verification",
                    studentBundle is null ? null : false);
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

    private static bool HasJsonBool(JsonElement source, string propertyName, bool expected)
    {
        return source.TryGetProperty(propertyName, out var property)
            && property.ValueKind == (expected ? JsonValueKind.True : JsonValueKind.False);
    }

    private static bool HasJsonLong(JsonElement source, string propertyName, long expected)
    {
        return source.TryGetProperty(propertyName, out var property)
            && property.TryGetInt64(out var actual)
            && actual == expected;
    }

    private static bool HasOptionalJsonLong(JsonElement source, string propertyName, long? expected)
    {
        if (expected is null)
        {
            return !source.TryGetProperty(propertyName, out var property)
                || property.ValueKind == JsonValueKind.Null;
        }
        return HasJsonLong(source, propertyName, expected.Value);
    }

    private static bool HasOptionalJsonBool(JsonElement source, string propertyName, bool? expected)
    {
        if (expected is null)
        {
            return !source.TryGetProperty(propertyName, out var property)
                || property.ValueKind == JsonValueKind.Null;
        }
        return HasJsonBool(source, propertyName, expected.Value);
    }
}

internal sealed record WhiteDocumentResult(
    int ResultSchemaVersion,
    string ResultSemanticsVersion,
    string DocumentType,
    string Source,
    string InferenceEngine,
    ImageSize ImageSize,
    WhiteDocumentRouteEvidence Route,
    WhiteDocumentOcrEvidence Ocr,
    IReadOnlyList<WhiteDocumentLine> Lines,
    DeviceResult Device,
    WhiteDocumentContractReferences ModelContracts,
    WhiteDocumentStageLatency StageLatencyMs,
    string[] Limitations);

internal sealed record WhiteDocumentRouteEvidence(
    string RequestedDocumentType,
    string ResolvedDocumentType,
    string Selector,
    bool CalibratedRouterAvailable,
    bool ReviewRequired,
    string ReviewReason);

internal sealed record WhiteDocumentOcrEvidence(
    string Role,
    string Pipeline,
    string Provider,
    string DeliveryPolicy,
    string StudentModelStatus,
    string FieldMappingStatus,
    string AggregateText,
    float? AggregateConfidence,
    int PassesDropScoreLineCount,
    int TotalLineCount,
    string? StudentProvider = null,
    int StudentComparisonLineCount = 0,
    int StudentNormalizedExactMatchLineCount = 0,
    string? StudentCropSource = null);

internal sealed record WhiteDocumentPoint(float X, float Y);

internal sealed record WhiteDocumentLine(
    int Index,
    string Text,
    float Confidence,
    bool PassesDropScore,
    IReadOnlyList<WhiteDocumentPoint> Quad,
    WhiteDocumentStudentLineEvidence? Student = null);

internal sealed record WhiteDocumentStudentLineEvidence(
    string Text,
    float Confidence,
    bool NormalizedExactMatch,
    string Provider,
    string DeliveryPolicy,
    string CropSource);

internal sealed record WhiteDocumentContractReferences(
    string Device,
    string DeviceSha256,
    string DeviceContractSha256,
    string OcrBundle,
    string OcrBundleContractSha256,
    string OcrSourceAuditContractSha256,
    string OcrDetector,
    string OcrDetectorSha256,
    string OcrClassifier,
    string OcrClassifierSha256,
    string OcrRecognizer,
    string OcrRecognizerSha256,
    string OcrDictionary,
    string OcrDictionarySha256,
    string RuntimeSource,
    bool ReopenedPathsAfterVerification,
    long OcrDictionarySnapshotSizeBytes,
    string OcrDictionarySnapshotSha256,
    string? WhiteStudentModel = null,
    string? WhiteStudentModelSha256 = null,
    long? WhiteStudentModelSnapshotSizeBytes = null,
    string? WhiteStudentCharset = null,
    string? WhiteStudentCharsetSha256 = null,
    long? WhiteStudentCharsetSnapshotSizeBytes = null,
    string? WhiteStudentContract = null,
    string? WhiteStudentContractSha256 = null,
    long? WhiteStudentContractSnapshotSizeBytes = null,
    string? WhiteStudentRuntimeSource = null,
    bool? WhiteStudentReopenedPathsAfterVerification = null);

internal sealed record WhiteDocumentManifestRecord(
    string Source,
    string Result,
    string Status,
    double? InferenceMs = null,
    WhiteDocumentStageLatency? StageLatencyMs = null);

internal sealed record WhiteDocumentInferenceSummary(
    string DocumentType,
    string RequestedDevice,
    string PaddleOcrProvider,
    int Input,
    int Written,
    int Skipped,
    int Errors,
    double TotalSeconds,
    LatencySummary InferenceLatencyMs,
    WhiteDocumentStageLatencySummary StageLatencyMs,
    string? WhiteStudentProvider = null);

internal sealed record WhiteDocumentStageLatency(
    double ImageLoad,
    double Device,
    double PaddleOcr,
    double ResultAssembly);

internal sealed record WhiteDocumentStageLatencySummary(
    LatencySummary ImageLoad,
    LatencySummary Device,
    LatencySummary PaddleOcr,
    LatencySummary ResultAssembly);
