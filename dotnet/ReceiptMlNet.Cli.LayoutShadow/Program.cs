using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

return LayoutShadowProgram.Run(args);

internal static class LayoutShadowProgram
{
    public const int ExpectedRecordCount = 339;
    public const string RecordKind = "receipt_ppocr_dotnet_cpu_layout_shadow_record_v1";
    public const string SummaryKind = "receipt_ppocr_dotnet_cpu_layout_shadow_summary_v1";
    public const string QuadCoordinateSpace = "max_side_1600_rectified_tl_tr_br_bl";
    public const string QuadNormalization = "x/(rectified_width-1),y/(rectified_height-1)";
    public const string ConfidenceSemantics = "ctc_emitted_character_mean";

    public static int Run(string[] args)
    {
        try
        {
            Run(LayoutShadowOptions.Parse(args));
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Paddle layout shadow failed: {error.Message}");
            return 1;
        }
    }

    private static void Run(LayoutShadowOptions options)
    {
        var selection = LayoutShadowInputContract.Load(
            options.InputList,
            options.InputListSha256);
        var output = LayoutShadowOutputContract.ResolveFreshOutput(options.Output);
        var bundle = PaddleOcrDeliveryBundle.LoadAndVerify(options.Bundle);
        var bundleEvidence = LayoutShadowBundleEvidence.From(bundle);
        LayoutShadowOutputContract.RequireDisjointFromBundle(
            output,
            bundleEvidence.Directory);

        using var engine = new PaddleOcrEngine(bundle, DeviceSetting.Parse("cpu"));
        if (!string.Equals(engine.ExecutionProvider, "cpu", StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Layout shadow must use the CPU execution provider, got {engine.ExecutionProvider}");
        }

        Directory.CreateDirectory(output.Parent);
        var stage = Path.Combine(
            output.Parent,
            $".{output.Name}.{Guid.NewGuid():N}.tmp");
        Directory.CreateDirectory(stage);
        try
        {
            var timings = new List<LayoutShadowTiming>(selection.Sources.Count);
            var sourceEvidence = new List<LayoutShadowSourceEvidence>(
                selection.Sources.Count);
            var recordsPath = Path.Combine(stage, "records.jsonl");
            using (var writer = new StreamWriter(
                recordsPath,
                append: false,
                encoding: new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)))
            {
                for (var index = 0; index < selection.Sources.Count; index++)
                {
                    var record = ReadOne(
                        index,
                        selection.Sources[index],
                        engine,
                        out var timing);
                    timings.Add(timing);
                    sourceEvidence.Add(new LayoutShadowSourceEvidence(
                        record.Source,
                        record.SourceImageSha256,
                        record.SourceImageSizeBytes));
                    writer.WriteLine(LayoutShadowJson.Serialize(record));
                    if ((index + 1) % 25 == 0 || index + 1 == selection.Sources.Count)
                    {
                        Console.WriteLine(
                            $"layout_shadow_cpu {index + 1}/{selection.Sources.Count}");
                    }
                }
            }

            // Re-read and verify every delivery byte before publishing the
            // evidence, so a package mutation during the run cannot silently
            // produce an unbound diagnostic result.
            var finalBundleEvidence = LayoutShadowBundleEvidence.From(
                PaddleOcrDeliveryBundle.LoadAndVerify(options.Bundle));
            if (finalBundleEvidence != bundleEvidence)
            {
                throw new InvalidOperationException(
                    "Paddle OCR delivery identity changed while layout shadow was running");
            }
            LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence);

            var recordsEvidence = LayoutShadowArtifactEvidence.From(
                recordsPath,
                relativePath: "records.jsonl");
            var summary = new LayoutShadowSummary(
                SchemaVersion: 1,
                Kind: SummaryKind,
                DiagnosticOnly: true,
                FormalDeliveryGate: false,
                CandidateWriteEnabled: false,
                ExpectedRecords: ExpectedRecordCount,
                Records: selection.Sources.Count,
                Errors: 0,
                ExecutionProvider: "cpu",
                Rectification: ReceiptRectifier.MaxSide1600Mode,
                QuadCoordinateSpace: QuadCoordinateSpace,
                QuadNormalization: QuadNormalization,
                ConfidenceSemantics: ConfidenceSemantics,
                PaddleDropScore: bundle.Settings.DropScore,
                InputList: new LayoutShadowInputEvidence(
                    selection.Path,
                    selection.Sha256,
                    selection.SizeBytes,
                    selection.Sources.Count),
                PaddleBundle: bundleEvidence,
                LatencyMs: LayoutShadowStageLatencySummary.From(timings),
                Artifacts: new LayoutShadowArtifacts(recordsEvidence));
            File.WriteAllText(
                Path.Combine(stage, "summary.json"),
                LayoutShadowJson.Serialize(summary) + Environment.NewLine,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            // Close the remaining publication window after summary creation.
            // These checks deliberately reread all bound bytes; the atomic
            // directory move is not allowed if the input list, any source
            // image, or any verified bundle component changed.
            LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence);
            var publicationBundleEvidence = LayoutShadowBundleEvidence.From(
                PaddleOcrDeliveryBundle.LoadAndVerify(options.Bundle));
            if (publicationBundleEvidence != bundleEvidence)
            {
                throw new InvalidOperationException(
                    "Paddle OCR delivery identity changed before layout shadow publication");
            }

            // Directory.Move is the only publication step. It also closes the
            // race where another process creates the target after preflight.
            Directory.Move(stage, output.FullPath);
        }
        catch
        {
            LayoutShadowOutputContract.DeleteOwnedStage(stage);
            throw;
        }

        Console.WriteLine(
            $"Wrote {ExpectedRecordCount} diagnostic-only Paddle layout record(s) to {output.FullPath}");
    }

    private static LayoutShadowRecord ReadOne(
        int index,
        string source,
        PaddleOcrEngine engine,
        out LayoutShadowTiming timing)
    {
        var totalStopwatch = Stopwatch.StartNew();
        var loadStopwatch = Stopwatch.StartNew();
        var imageBytes = File.ReadAllBytes(source);
        var imageSha256 = LayoutShadowHash.Sha256(imageBytes);
        using var imageStream = new MemoryStream(imageBytes, writable: false);
        using var sourceImage = Image.Load<Rgb24>(imageStream);
        sourceImage.Mutate(context => context.AutoOrient());
        loadStopwatch.Stop();

        var rectificationStopwatch = Stopwatch.StartNew();
        using var rectification = ReceiptRectifier.Rectify(
            sourceImage,
            ReceiptRectifier.MaxSide1600Mode);
        rectificationStopwatch.Stop();

        var ocrStopwatch = Stopwatch.StartNew();
        var read = engine.RecognizeLayoutDiagnostic(rectification.Image);
        ocrStopwatch.Stop();
        totalStopwatch.Stop();

        timing = new LayoutShadowTiming(
            RoundMilliseconds(loadStopwatch),
            RoundMilliseconds(rectificationStopwatch),
            RoundMilliseconds(ocrStopwatch),
            RoundMilliseconds(totalStopwatch));
        var geometry = rectification.Geometry();
        var lines = BuildLayoutLines(
            read,
            rectification.Image.Width,
            rectification.Image.Height);

        return new LayoutShadowRecord(
            SchemaVersion: 1,
            Kind: RecordKind,
            DiagnosticOnly: true,
            FormalDeliveryGate: false,
            CandidateWriteEnabled: false,
            Index: index,
            Source: source,
            SourceImageSha256: imageSha256,
            SourceImageSizeBytes: imageBytes.LongLength,
            ExecutionProvider: "cpu",
            Geometry: geometry,
            QuadCoordinateSpace: QuadCoordinateSpace,
            QuadNormalization: QuadNormalization,
            ConfidenceSemantics: ConfidenceSemantics,
            AcceptedText: read.Text,
            AcceptedConfidence: read.Confidence,
            AcceptedLineCount: read.AcceptedLines.Count,
            RawLineCount: lines.Count,
            Lines: lines,
            TimingMs: timing);
    }

    internal static IReadOnlyList<LayoutShadowLine> BuildLayoutLines(
        PaddleOcrLayoutReadResult read,
        int rectifiedWidth,
        int rectifiedHeight)
    {
        ArgumentNullException.ThrowIfNull(read);
        if (rectifiedWidth < 2 || rectifiedHeight < 2)
        {
            throw new InvalidOperationException(
                "Layout shadow rectified dimensions must both be at least two pixels");
        }
        var widthMaximum = rectifiedWidth - 1.0f;
        var heightMaximum = rectifiedHeight - 1.0f;
        var acceptedIndex = 0;
        var lines = new LayoutShadowLine[read.Lines.Count];
        for (var lineIndex = 0; lineIndex < read.Lines.Count; lineIndex++)
        {
            var line = read.Lines[lineIndex];
            if (!float.IsFinite(line.Confidence)
                || line.Confidence < 0.0f
                || line.Confidence > 1.0f)
            {
                throw new InvalidOperationException(
                    $"Layout shadow line {lineIndex} confidence is outside [0, 1]");
            }
            if (line.Quad.Count != 4
                || line.Quad.Any(point =>
                    !float.IsFinite(point.X)
                    || !float.IsFinite(point.Y)
                    || point.X < 0.0f
                    || point.X > widthMaximum
                    || point.Y < 0.0f
                    || point.Y > heightMaximum))
            {
                throw new InvalidOperationException(
                    $"Layout shadow line {lineIndex} quad is outside rectified image bounds");
            }
            if (line.PassesDropScore)
            {
                if (acceptedIndex >= read.AcceptedLines.Count)
                {
                    throw new InvalidOperationException(
                        $"Layout shadow line {lineIndex} has no accepted CTC output");
                }
                var accepted = read.AcceptedLines[acceptedIndex];
                if (!string.Equals(accepted.Text, line.Text, StringComparison.Ordinal)
                    || BitConverter.SingleToInt32Bits(accepted.Confidence)
                        != BitConverter.SingleToInt32Bits(line.Confidence))
                {
                    throw new InvalidOperationException(
                        $"Layout shadow line {lineIndex} is not index-bound to accepted CTC output");
                }
                acceptedIndex++;
            }
            lines[lineIndex] = new LayoutShadowLine(
                Index: lineIndex,
                Text: line.Text,
                Confidence: line.Confidence,
                PassesDropScore: line.PassesDropScore,
                QuadRectified: line.Quad
                    .Select(point => new[] { point.X, point.Y })
                    .ToArray(),
                QuadRectifiedNormalized: line.Quad
                    .Select(point => new[]
                    {
                        point.X / widthMaximum,
                        point.Y / heightMaximum,
                    })
                    .ToArray());
        }
        if (acceptedIndex != read.AcceptedLines.Count)
        {
            throw new InvalidOperationException(
                "Layout shadow accepted CTC line count differs from DB line projection");
        }
        return Array.AsReadOnly(lines);
    }

    private static double RoundMilliseconds(Stopwatch stopwatch)
    {
        return Math.Round(stopwatch.Elapsed.TotalMilliseconds, 4, MidpointRounding.ToEven);
    }
}

internal sealed record LayoutShadowOptions(
    string Bundle,
    string InputList,
    string InputListSha256,
    string Output)
{
    public static LayoutShadowOptions Parse(string[] args)
    {
        string? bundle = null;
        string? inputList = null;
        string? inputListSha256 = null;
        string? output = null;
        for (var index = 0; index < args.Length; index++)
        {
            var name = args[index];
            if (index + 1 >= args.Length)
            {
                throw new InvalidOperationException($"Missing value for layout shadow argument {name}");
            }
            var value = args[++index];
            switch (name)
            {
                case "--bundle": bundle = SetOnce(bundle, value, name); break;
                case "--input-list": inputList = SetOnce(inputList, value, name); break;
                case "--input-list-sha256": inputListSha256 = SetOnce(inputListSha256, value, name); break;
                case "--output": output = SetOnce(output, value, name); break;
                default: throw new InvalidOperationException($"Unknown layout shadow argument: {name}");
            }
        }
        if (string.IsNullOrWhiteSpace(bundle)
            || string.IsNullOrWhiteSpace(inputList)
            || string.IsNullOrWhiteSpace(inputListSha256)
            || string.IsNullOrWhiteSpace(output))
        {
            throw new InvalidOperationException(
                "Usage: --bundle <verified-delivery> --input-list <frozen-339.txt> "
                + "--input-list-sha256 <lowercase-sha256> --output <fresh-directory>");
        }
        return new LayoutShadowOptions(bundle, inputList, inputListSha256, output);
    }

    private static string SetOnce(string? current, string value, string name)
    {
        if (current is not null)
        {
            throw new InvalidOperationException($"Duplicate layout shadow argument: {name}");
        }
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException($"Layout shadow argument {name} must not be blank");
        }
        return value;
    }
}

internal static class LayoutShadowInputContract
{
    private static readonly HashSet<string> ImageExtensions = new(
        [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"],
        StringComparer.OrdinalIgnoreCase);
    private static readonly UTF8Encoding StrictUtf8 = new(
        encoderShouldEmitUTF8Identifier: false,
        throwOnInvalidBytes: true);

    public static LayoutShadowInputSelection Load(string inputList, string expectedSha256)
    {
        LayoutShadowHash.RequireLowerSha256(expectedSha256, "--input-list-sha256");
        var path = Path.GetFullPath(inputList);
        if (!File.Exists(path))
        {
            throw new InvalidOperationException($"Layout shadow input list does not exist: {path}");
        }
        LayoutShadowFileContract.RequireRegularNonReparseFile(
            path,
            "Layout shadow input list");
        var bytes = File.ReadAllBytes(path);
        var actualSha256 = LayoutShadowHash.Sha256(bytes);
        if (!string.Equals(actualSha256, expectedSha256, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Layout shadow input-list SHA-256 differs: expected={expectedSha256} actual={actualSha256}");
        }

        var text = StrictUtf8.GetString(bytes);
        if (text.Length > 0 && text[0] == '\uFEFF')
        {
            text = text[1..];
        }
        var comparer = OperatingSystem.IsWindows()
            ? StringComparer.OrdinalIgnoreCase
            : StringComparer.Ordinal;
        var seen = new HashSet<string>(comparer);
        var sources = new List<string>(LayoutShadowProgram.ExpectedRecordCount);
        using var reader = new StringReader(text);
        var lineNumber = 0;
        while (reader.ReadLine() is { } rawLine)
        {
            lineNumber++;
            if (rawLine.Length == 0)
            {
                throw new InvalidOperationException(
                    $"Layout shadow input list contains a blank line at {path}:{lineNumber}");
            }
            if (!string.Equals(rawLine, rawLine.Trim(), StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Layout shadow input list contains surrounding whitespace at {path}:{lineNumber}");
            }
            if (rawLine[0] == '#')
            {
                throw new InvalidOperationException(
                    $"Layout shadow input list must not contain comments at {path}:{lineNumber}");
            }
            if (!Path.IsPathFullyQualified(rawLine))
            {
                throw new InvalidOperationException(
                    $"Layout shadow source must be an absolute path at {path}:{lineNumber}");
            }

            string source;
            try
            {
                source = Path.GetFullPath(rawLine);
            }
            catch (Exception error) when (
                error is ArgumentException or NotSupportedException or PathTooLongException)
            {
                throw new InvalidOperationException(
                    $"Invalid layout shadow source at {path}:{lineNumber}: {error.Message}",
                    error);
            }
            if (!File.Exists(source))
            {
                throw new InvalidOperationException(
                    $"Layout shadow source does not exist at {path}:{lineNumber}: {source}");
            }
            LayoutShadowFileContract.RequireRegularNonReparseFile(
                source,
                $"Layout shadow source at {path}:{lineNumber}");
            if (!ImageExtensions.Contains(Path.GetExtension(source)))
            {
                throw new InvalidOperationException(
                    $"Unsupported layout shadow image at {path}:{lineNumber}: {source}");
            }
            if (!seen.Add(source))
            {
                throw new InvalidOperationException(
                    $"Duplicate layout shadow source at {path}:{lineNumber}: {source}");
            }
            sources.Add(source);
        }

        if (sources.Count != LayoutShadowProgram.ExpectedRecordCount)
        {
            throw new InvalidOperationException(
                $"Layout shadow requires exactly {LayoutShadowProgram.ExpectedRecordCount} sources, found {sources.Count}");
        }
        return new LayoutShadowInputSelection(
            path,
            actualSha256,
            bytes.LongLength,
            sources.AsReadOnly());
    }

    public static void VerifyUnchanged(
        LayoutShadowInputSelection selection,
        IReadOnlyList<LayoutShadowSourceEvidence> sourceEvidence)
    {
        ArgumentNullException.ThrowIfNull(selection);
        ArgumentNullException.ThrowIfNull(sourceEvidence);
        LayoutShadowFileContract.RequireRegularNonReparseFile(
            selection.Path,
            "Layout shadow input list");
        var currentList = LayoutShadowHash.Sha256FileEvidence(selection.Path);
        if (currentList.SizeBytes != selection.SizeBytes
            || !string.Equals(
                currentList.Sha256,
                selection.Sha256,
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "Layout shadow input list changed while the diagnostic was running");
        }
        if (sourceEvidence.Count != selection.Sources.Count)
        {
            throw new InvalidOperationException(
                "Layout shadow source evidence count differs from the frozen input list");
        }
        for (var index = 0; index < selection.Sources.Count; index++)
        {
            var source = selection.Sources[index];
            var expected = sourceEvidence[index];
            if (!string.Equals(expected.Path, source, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Layout shadow source evidence order differs at index {index}");
            }
            LayoutShadowFileContract.RequireRegularNonReparseFile(
                source,
                $"Layout shadow source at index {index}");
            var current = LayoutShadowHash.Sha256FileEvidence(source);
            if (current.SizeBytes != expected.SizeBytes
                || !string.Equals(
                    current.Sha256,
                    expected.Sha256,
                    StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Layout shadow source changed while the diagnostic was running: {source}");
            }
        }
    }
}

internal static class LayoutShadowOutputContract
{
    public static LayoutShadowOutput ResolveFreshOutput(string output)
    {
        var fullPath = Path.GetFullPath(output)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var name = Path.GetFileName(fullPath);
        var parent = Path.GetDirectoryName(fullPath);
        if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(parent))
        {
            throw new InvalidOperationException("Layout shadow output must be a non-root directory path");
        }
        if (File.Exists(fullPath) || Directory.Exists(fullPath))
        {
            throw new InvalidOperationException($"Refusing to overwrite layout shadow output: {fullPath}");
        }
        return new LayoutShadowOutput(fullPath, parent, name);
    }

    public static void RequireDisjointFromBundle(
        LayoutShadowOutput output,
        string bundleDirectory)
    {
        ArgumentNullException.ThrowIfNull(output);
        var bundle = Path.GetFullPath(bundleDirectory);
        if (IsWithin(output.FullPath, bundle) || IsWithin(bundle, output.FullPath))
        {
            throw new InvalidOperationException(
                "Layout shadow output must be disjoint from the verified Paddle OCR delivery directory");
        }
    }

    public static void DeleteOwnedStage(string stage)
    {
        try
        {
            if (Directory.Exists(stage))
            {
                Directory.Delete(stage, recursive: true);
            }
        }
        catch
        {
            // Preserve the inference/publication error. The randomly named
            // stage can be inspected and removed manually if cleanup fails.
        }
    }


    private static bool IsWithin(string candidate, string root)
    {
        var relative = Path.GetRelativePath(root, candidate);
        return !Path.IsPathRooted(relative)
            && !string.Equals(relative, "..", StringComparison.Ordinal)
            && !relative.StartsWith(
                ".." + Path.DirectorySeparatorChar,
                StringComparison.Ordinal)
            && !relative.StartsWith(
                ".." + Path.AltDirectorySeparatorChar,
                StringComparison.Ordinal);
    }
}

internal static class LayoutShadowFileContract
{
    public static void RequireRegularNonReparseDirectory(
        string path,
        string description)
    {
        var attributes = File.GetAttributes(path);
        if ((attributes & FileAttributes.Directory) == 0
            || (attributes & FileAttributes.ReparsePoint) != 0)
        {
            throw new InvalidOperationException(
                $"{description} must be a regular non-reparse directory: {path}");
        }
    }

    public static void RequireRegularNonReparseFile(string path, string description)
    {
        var attributes = File.GetAttributes(path);
        if ((attributes & FileAttributes.Directory) != 0
            || (attributes & FileAttributes.ReparsePoint) != 0)
        {
            throw new InvalidOperationException(
                $"{description} must be a regular non-reparse file: {path}");
        }
    }
}

internal static class LayoutShadowHash
{
    public static string Sha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    public static LayoutShadowFileHashEvidence Sha256FileEvidence(string path)
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read);
        var sizeBytes = stream.Length;
        var sha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        return new LayoutShadowFileHashEvidence(sha256, sizeBytes);
    }

    public static void RequireLowerSha256(string value, string description)
    {
        if (value.Length != 64 || value.Any(character => !IsLowerHex(character)))
        {
            throw new InvalidOperationException(
                $"{description} must be exactly 64 lowercase hexadecimal characters");
        }
    }

    private static bool IsLowerHex(char character)
    {
        return (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f');
    }
}

internal static class LayoutShadowJson
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        WriteIndented = false,
    };

    public static string Serialize<T>(T value)
    {
        return JsonSerializer.Serialize(value, Options);
    }
}

internal sealed record LayoutShadowInputSelection(
    string Path,
    string Sha256,
    long SizeBytes,
    IReadOnlyList<string> Sources);

internal sealed record LayoutShadowSourceEvidence(
    string Path,
    string Sha256,
    long SizeBytes);

internal sealed record LayoutShadowFileHashEvidence(
    string Sha256,
    long SizeBytes);

internal sealed record LayoutShadowOutput(string FullPath, string Parent, string Name);

internal sealed record LayoutShadowTiming(
    double ImageLoad,
    double Rectification,
    double LayoutOcr,
    double Total);

internal sealed record LayoutShadowLine(
    int Index,
    string Text,
    float Confidence,
    bool PassesDropScore,
    float[][] QuadRectified,
    float[][] QuadRectifiedNormalized);

internal sealed record LayoutShadowRecord(
    int SchemaVersion,
    string Kind,
    bool DiagnosticOnly,
    bool FormalDeliveryGate,
    bool CandidateWriteEnabled,
    int Index,
    string Source,
    string SourceImageSha256,
    long SourceImageSizeBytes,
    string ExecutionProvider,
    RectificationGeometry Geometry,
    string QuadCoordinateSpace,
    string QuadNormalization,
    string ConfidenceSemantics,
    string AcceptedText,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] float? AcceptedConfidence,
    int AcceptedLineCount,
    int RawLineCount,
    IReadOnlyList<LayoutShadowLine> Lines,
    LayoutShadowTiming TimingMs);

internal sealed record LayoutShadowInputEvidence(
    string Path,
    string Sha256,
    long SizeBytes,
    int Records);

internal sealed record LayoutShadowFileEvidence(
    string RelativePath,
    string Sha256,
    long SizeBytes);

internal sealed record LayoutShadowBundleEvidence(
    string Directory,
    string ContractPath,
    string ContractSha256,
    string SourceAuditContractSha256,
    long PackageSizeBytes,
    LayoutShadowFileEvidence Detector,
    LayoutShadowFileEvidence Classifier,
    LayoutShadowFileEvidence Recognizer,
    LayoutShadowFileEvidence Dictionary)
{
    public static LayoutShadowBundleEvidence From(PaddleOcrDeliveryBundle bundle)
    {
        LayoutShadowFileContract.RequireRegularNonReparseDirectory(
            bundle.BundleDirectory,
            "Paddle OCR delivery directory");
        LayoutShadowFileContract.RequireRegularNonReparseFile(
            bundle.ContractPath,
            "Paddle OCR delivery contract");
        foreach (var file in new[]
        {
            bundle.DetModel.File,
            bundle.ClsModel.File,
            bundle.RecModel.File,
            bundle.Dictionary,
        })
        {
            LayoutShadowFileContract.RequireRegularNonReparseFile(
                file.FullPath,
                $"Paddle OCR delivery component {file.RelativePath}");
        }
        return new LayoutShadowBundleEvidence(
            bundle.BundleDirectory,
            bundle.ContractPath,
            bundle.ContractSha256,
            bundle.SourceAuditContractSha256,
            bundle.PackageSizeBytes,
            FromFile(bundle.DetModel.File),
            FromFile(bundle.ClsModel.File),
            FromFile(bundle.RecModel.File),
            FromFile(bundle.Dictionary));
    }

    private static LayoutShadowFileEvidence FromFile(PaddleOcrFileRecord file)
    {
        return new LayoutShadowFileEvidence(
            file.RelativePath,
            file.Sha256,
            file.SizeBytes);
    }
}

internal sealed record LayoutShadowArtifactEvidence(
    string RelativePath,
    string Sha256,
    long SizeBytes)
{
    public static LayoutShadowArtifactEvidence From(string path, string relativePath)
    {
        var identity = LayoutShadowHash.Sha256FileEvidence(path);
        return new LayoutShadowArtifactEvidence(
            relativePath,
            identity.Sha256,
            identity.SizeBytes);
    }
}

internal sealed record LayoutShadowLatencyDistribution(
    int Count,
    double Mean,
    double P50,
    double P95,
    double P99,
    double Max)
{
    public static LayoutShadowLatencyDistribution From(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(value => value).ToArray();
        if (sorted.Length == 0 || sorted.Any(value => !double.IsFinite(value) || value < 0.0))
        {
            throw new InvalidOperationException("Layout shadow latency evidence is empty or non-finite");
        }
        return new LayoutShadowLatencyDistribution(
            sorted.Length,
            Round(sorted.Average()),
            Round(Percentile(sorted, 0.50)),
            Round(Percentile(sorted, 0.95)),
            Round(Percentile(sorted, 0.99)),
            Round(sorted[^1]));
    }

    private static double Percentile(IReadOnlyList<double> sorted, double quantile)
    {
        var position = (sorted.Count - 1) * quantile;
        var lower = (int)Math.Floor(position);
        var upper = (int)Math.Ceiling(position);
        return lower == upper
            ? sorted[lower]
            : sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static double Round(double value)
    {
        return Math.Round(value, 4, MidpointRounding.ToEven);
    }
}

internal sealed record LayoutShadowStageLatencySummary(
    LayoutShadowLatencyDistribution ImageLoad,
    LayoutShadowLatencyDistribution Rectification,
    LayoutShadowLatencyDistribution LayoutOcr,
    LayoutShadowLatencyDistribution Total)
{
    public static LayoutShadowStageLatencySummary From(
        IReadOnlyCollection<LayoutShadowTiming> timings)
    {
        return new LayoutShadowStageLatencySummary(
            LayoutShadowLatencyDistribution.From(timings.Select(value => value.ImageLoad)),
            LayoutShadowLatencyDistribution.From(timings.Select(value => value.Rectification)),
            LayoutShadowLatencyDistribution.From(timings.Select(value => value.LayoutOcr)),
            LayoutShadowLatencyDistribution.From(timings.Select(value => value.Total)));
    }
}

internal sealed record LayoutShadowArtifacts(LayoutShadowArtifactEvidence RecordsJsonl);

internal sealed record LayoutShadowSummary(
    int SchemaVersion,
    string Kind,
    bool DiagnosticOnly,
    bool FormalDeliveryGate,
    bool CandidateWriteEnabled,
    int ExpectedRecords,
    int Records,
    int Errors,
    string ExecutionProvider,
    string Rectification,
    string QuadCoordinateSpace,
    string QuadNormalization,
    string ConfidenceSemantics,
    float PaddleDropScore,
    LayoutShadowInputEvidence InputList,
    LayoutShadowBundleEvidence PaddleBundle,
    LayoutShadowStageLatencySummary LatencyMs,
    LayoutShadowArtifacts Artifacts);
