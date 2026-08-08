using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

return RecipientDerivedCropShadowProgram.Run(args);

internal static class RecipientDerivedCropShadowProgram
{
    public const int ExpectedRecords = 63;
    public const string PlanSummaryKind = "receipt_mlnet_recipient_derived_crop_plan_summary_v1";
    public const string PlanRecordKind = "receipt_mlnet_recipient_derived_crop_plan_record_v1";
    public const string SummaryKind = "receipt_mlnet_recipient_derived_crop_layout_summary_v1";
    public const string RecordKind = "receipt_mlnet_recipient_derived_crop_layout_record_v1";
    public const string Crop4 = "crop4_interrow_value_corridor";
    public const string Crop5 = "crop5_recipient_value_core";
    public const string PlanRectification = "max_side_1600";
    public const string QuadCoordinateSpace = "crop_tl_tr_br_bl_and_max_side_1600_rectified";
    public const string ConfidenceSemantics = "ctc_emitted_character_mean";

    public static int Run(string[] args)
    {
        try
        {
            Run(RecipientDerivedCropShadowOptions.Parse(args));
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Recipient derived-crop shadow failed: {error.Message}");
            return 1;
        }
    }

    private static void Run(RecipientDerivedCropShadowOptions options)
    {
        var plan = RecipientDerivedCropPlanContract.Load(
            options.PlanDirectory,
            options.PlanSummarySha256);
        var output = RecipientDerivedCropOutputContract.ResolveFreshOutput(options.Output);
        var sourceBundle = PaddleOcrDeliveryBundle.LoadAndVerify(options.Bundle);
        var bundleEvidence = RecipientDerivedCropBundleEvidence.From(sourceBundle);
        RecipientDerivedCropOutputContract.RequireDisjoint(
            output,
            plan.Directory,
            "derived-crop plan directory");
        RecipientDerivedCropOutputContract.RequireDisjoint(
            output,
            bundleEvidence.Directory,
            "verified Paddle OCR delivery directory");

        Directory.CreateDirectory(output.Parent);
        RecipientDerivedCropOutputContract.VerifyPublicationParent(output);
        var stage = Path.Combine(output.Parent, $".{output.Name}.{Guid.NewGuid():N}.tmp");
        try
        {
            Directory.CreateDirectory(stage);
            RecipientDerivedCropOutputContract.VerifyOwnedStage(output, stage);
            var privateBundleDirectory = Path.Combine(stage, ".paddle-bundle-snapshot");
            RecipientDerivedCropBundleSnapshot.Create(sourceBundle, privateBundleDirectory);
            var bundle = PaddleOcrDeliveryBundle.LoadAndVerify(privateBundleDirectory);
            var privateBundleEvidence = RecipientDerivedCropBundleEvidence.From(bundle);
            if (!privateBundleEvidence.ContentEquals(bundleEvidence))
            {
                throw new InvalidOperationException(
                    "Private Paddle OCR bundle snapshot differs from the verified source delivery");
            }
            var cpuModelSnapshot = PaddleOcrCpuModelSnapshot.Create(
                bundle,
                RecipientDerivedCropFileContract.ReadRegularFile(
                    bundle.DetModel.File.FullPath,
                    "Private Paddle OCR detector model snapshot"),
                RecipientDerivedCropFileContract.ReadRegularFile(
                    bundle.ClsModel.File.FullPath,
                    "Private Paddle OCR classifier model snapshot"),
                RecipientDerivedCropFileContract.ReadRegularFile(
                    bundle.RecModel.File.FullPath,
                    "Private Paddle OCR recognizer model snapshot"));
            var timings = new List<RecipientDerivedCropTiming>(ExpectedRecords);
            var recordsPath = Path.Combine(stage, "records.jsonl");
            using (var engine = new PaddleOcrEngine(bundle, cpuModelSnapshot))
            {
                if (!string.Equals(engine.ExecutionProvider, "cpu", StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"Recipient derived-crop shadow must use the CPU execution provider, got {engine.ExecutionProvider}");
                }
                using var writer = new StreamWriter(
                    recordsPath,
                    append: false,
                    encoding: new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
                for (var index = 0; index < plan.Records.Count; index++)
                {
                    var record = ReadOne(
                        index,
                        plan.Records[index],
                        engine,
                        bundle.Settings.DropScore);
                    timings.Add(record.TimingMs);
                    writer.WriteLine(RecipientDerivedCropJson.Serialize(record));
                    if ((index + 1) % 10 == 0
                        || index + 1 == plan.Records.Count)
                    {
                        Console.WriteLine($"recipient_derived_crop_cpu {index + 1}/{plan.Records.Count}");
                    }
                }
            }
            Directory.Delete(privateBundleDirectory, recursive: true);
            RecipientDerivedCropOutputContract.VerifyOwnedStage(output, stage);

            VerifyClosingEvidence(plan, options.Bundle, bundleEvidence);
            var recordsEvidence = RecipientDerivedCropArtifactEvidence.From(
                recordsPath,
                path: "records.jsonl",
                records: ExpectedRecords);
            var summary = new RecipientDerivedCropSummary(
                SchemaVersion: 1,
                Kind: SummaryKind,
                DiagnosticOnly: true,
                FormalDeliveryGate: false,
                CandidateWriteEnabled: false,
                ProductionOutputChanged: false,
                AccuracyClaimed: false,
                TruthUsedForCandidateSelection: false,
                OcrRerun: true,
                ExpectedRecords: ExpectedRecords,
                Records: ExpectedRecords,
                Errors: 0,
                ExecutionProvider: "cpu",
                Rectification: PlanRectification,
                CropNames: [Crop4, Crop5],
                QuadCoordinateSpace: QuadCoordinateSpace,
                ConfidenceSemantics: ConfidenceSemantics,
                PaddleDropScore: bundle.Settings.DropScore,
                InputPlan: RecipientDerivedCropInputPlanEvidence.From(plan),
                PaddleBundle: bundleEvidence,
                LatencyMs: RecipientDerivedCropLatencySummary.From(timings),
                Artifacts: new RecipientDerivedCropArtifacts(recordsEvidence));
            File.WriteAllText(
                Path.Combine(stage, "summary.json"),
                RecipientDerivedCropJson.Serialize(summary) + Environment.NewLine,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            // The publication check deliberately repeats every plan, source,
            // and model identity after summary creation. Directory.Move is the
            // only visible publication step.
            VerifyClosingEvidence(plan, options.Bundle, bundleEvidence);
            RecipientDerivedCropOutputContract.RequireDisjoint(
                output,
                plan.Directory,
                "derived-crop plan directory");
            RecipientDerivedCropOutputContract.RequireDisjoint(
                output,
                bundleEvidence.Directory,
                "verified Paddle OCR delivery directory");
            RecipientDerivedCropOutputContract.VerifyOwnedStage(output, stage);
            Directory.Move(stage, output.FullPath);
        }
        catch
        {
            RecipientDerivedCropOutputContract.DeleteOwnedStage(stage);
            throw;
        }

        Console.WriteLine(
            $"Wrote {ExpectedRecords} diagnostic-only recipient crop4/crop5 layout record(s) to {output.FullPath}");
    }

    private static void VerifyClosingEvidence(
        RecipientDerivedCropPlanSelection plan,
        string bundlePath,
        RecipientDerivedCropBundleEvidence expectedBundle)
    {
        RecipientDerivedCropPlanContract.VerifyUnchanged(plan);
        var currentBundle = RecipientDerivedCropBundleEvidence.From(
            PaddleOcrDeliveryBundle.LoadAndVerify(bundlePath));
        if (currentBundle != expectedBundle)
        {
            throw new InvalidOperationException(
                "Verified Paddle OCR delivery identity changed while recipient derived-crop shadow was running");
        }
    }

    private static RecipientDerivedCropRecord ReadOne(
        int index,
        RecipientDerivedCropPlanRecord plan,
        PaddleOcrEngine engine,
        float dropScore)
    {
        var totalStopwatch = Stopwatch.StartNew();
        var loadStopwatch = Stopwatch.StartNew();
        var imageBytes = RecipientDerivedCropFileContract.ReadRegularFile(
            plan.Source,
            $"source image at plan index {index}");
        var sourceIdentity = RecipientDerivedCropFileIdentity.FromBytes(plan.Source, imageBytes);
        if (!sourceIdentity.ContentEquals(plan.SourceImage))
        {
            throw new InvalidOperationException(
                $"Source image differs from the frozen derived-crop plan: {plan.Source}");
        }
        using var imageStream = new MemoryStream(imageBytes, writable: false);
        using var sourceImage = Image.Load<Rgb24>(imageStream);
        sourceImage.Mutate(context => context.AutoOrient());
        loadStopwatch.Stop();

        var rectificationStopwatch = Stopwatch.StartNew();
        using var rectification = ReceiptRectifier.Rectify(
            sourceImage,
            ReceiptRectifier.MaxSide1600Mode);
        rectificationStopwatch.Stop();
        if (rectification.Image.Width != plan.RectifiedSize.Width
            || rectification.Image.Height != plan.RectifiedSize.Height)
        {
            throw new InvalidOperationException(
                $"Rectified image size differs from frozen plan for {plan.Source}: "
                + $"expected={plan.RectifiedSize.Width}x{plan.RectifiedSize.Height} "
                + $"actual={rectification.Image.Width}x{rectification.Image.Height}");
        }

        var crops = new List<RecipientDerivedCropLayout>(2);
        var cropOcrMilliseconds = new List<double>(2);
        foreach (var cropPlan in plan.Crops)
        {
            using var cropImage = rectification.Image.Clone(context => context.Crop(
                new Rectangle(
                    cropPlan.Box.Left,
                    cropPlan.Box.Top,
                    cropPlan.Box.Width,
                    cropPlan.Box.Height)));
            var ocrStopwatch = Stopwatch.StartNew();
            var read = engine.RecognizeLayoutDiagnostic(cropImage);
            ocrStopwatch.Stop();
            cropOcrMilliseconds.Add(RoundMilliseconds(ocrStopwatch));
            crops.Add(new RecipientDerivedCropLayout(
                Name: cropPlan.Name,
                RectifiedBox: cropPlan.Box.ToArray(),
                Width: cropPlan.Box.Width,
                Height: cropPlan.Box.Height,
                Lines: BuildLines(read, cropPlan, dropScore)));
        }
        totalStopwatch.Stop();

        return new RecipientDerivedCropRecord(
            SchemaVersion: 1,
            Kind: RecordKind,
            DiagnosticOnly: true,
            FormalDeliveryGate: false,
            CandidateWriteEnabled: false,
            ProductionOutputChanged: false,
            Index: index,
            Source: plan.Source,
            SourceImageSha256: sourceIdentity.Sha256,
            SourceImageSizeBytes: sourceIdentity.SizeBytes,
            ExecutionProvider: "cpu",
            Rectification: PlanRectification,
            RectifiedSize: plan.RectifiedSize,
            PlanId: plan.PlanId,
            QuadCoordinateSpace: QuadCoordinateSpace,
            ConfidenceSemantics: ConfidenceSemantics,
            Crops: crops.AsReadOnly(),
            TimingMs: new RecipientDerivedCropTiming(
                ImageLoad: RoundMilliseconds(loadStopwatch),
                Rectification: RoundMilliseconds(rectificationStopwatch),
                Crop4LayoutOcr: cropOcrMilliseconds[0],
                Crop5LayoutOcr: cropOcrMilliseconds[1],
                Total: RoundMilliseconds(totalStopwatch)));
    }

    internal static IReadOnlyList<RecipientDerivedCropLine> BuildLines(
        PaddleOcrLayoutReadResult read,
        RecipientDerivedCropPlanCrop crop,
        float dropScore)
    {
        ArgumentNullException.ThrowIfNull(read);
        ArgumentNullException.ThrowIfNull(crop);
        if (!float.IsFinite(dropScore) || dropScore < 0.0f || dropScore > 1.0f)
        {
            throw new InvalidOperationException("Paddle OCR drop score must be within [0, 1]");
        }
        var lines = new RecipientDerivedCropLine[read.Lines.Count];
        for (var lineIndex = 0; lineIndex < read.Lines.Count; lineIndex++)
        {
            var line = read.Lines[lineIndex];
            if (line.Text is null)
            {
                throw new InvalidOperationException(
                    $"Recipient derived-crop line {lineIndex} text is null");
            }
            if (!float.IsFinite(line.Confidence)
                || line.Confidence < 0.0f
                || line.Confidence > 1.0f)
            {
                throw new InvalidOperationException(
                    $"Recipient derived-crop line {lineIndex} confidence is outside [0, 1]");
            }
            var expectedPass = line.Confidence >= dropScore;
            if (line.PassesDropScore != expectedPass)
            {
                throw new InvalidOperationException(
                    $"Recipient derived-crop line {lineIndex} drop-score flag differs from verified bundle");
            }
            if (line.Quad.Count != 4)
            {
                throw new InvalidOperationException(
                    $"Recipient derived-crop line {lineIndex} is not a four-point quadrilateral");
            }
            RequireOrderedConvexQuad(line.Quad, lineIndex);
            var quadCrop = new float[4][];
            var quadRectified = new float[4][];
            for (var pointIndex = 0; pointIndex < 4; pointIndex++)
            {
                var point = line.Quad[pointIndex];
                if (!float.IsFinite(point.X)
                    || !float.IsFinite(point.Y)
                    || point.X < -1.0f
                    || point.X > crop.Box.Width + 1.0f
                    || point.Y < -1.0f
                    || point.Y > crop.Box.Height + 1.0f)
                {
                    throw new InvalidOperationException(
                        $"Recipient derived-crop line {lineIndex} quad escapes crop bounds");
                }
                quadCrop[pointIndex] = [point.X, point.Y];
                quadRectified[pointIndex] = [
                    point.X + crop.Box.Left,
                    point.Y + crop.Box.Top,
                ];
            }
            lines[lineIndex] = new RecipientDerivedCropLine(
                Index: lineIndex,
                Text: line.Text,
                Confidence: line.Confidence,
                PassesDropScore: line.PassesDropScore,
                QuadCrop: quadCrop,
                QuadRectified: quadRectified);
        }
        return Array.AsReadOnly(lines);
    }

    private static void RequireOrderedConvexQuad(
        IReadOnlyList<PaddleOcrLayoutPoint> quad,
        int lineIndex)
    {
        static double Cross(
            PaddleOcrLayoutPoint first,
            PaddleOcrLayoutPoint second,
            PaddleOcrLayoutPoint third)
        {
            return (second.X - first.X) * (third.Y - second.Y)
                - (second.Y - first.Y) * (third.X - second.X);
        }

        for (var pointIndex = 0; pointIndex < 4; pointIndex++)
        {
            var cross = Cross(
                quad[pointIndex],
                quad[(pointIndex + 1) % 4],
                quad[(pointIndex + 2) % 4]);
            if (!double.IsFinite(cross) || cross <= 1e-3)
            {
                throw new InvalidOperationException(
                    $"Recipient derived-crop line {lineIndex} quad is degenerate, non-convex, or not TL/TR/BR/BL");
            }
        }
        var topY = (quad[0].Y + quad[1].Y) * 0.5;
        var bottomY = (quad[2].Y + quad[3].Y) * 0.5;
        var leftX = (quad[0].X + quad[3].X) * 0.5;
        var rightX = (quad[1].X + quad[2].X) * 0.5;
        if (!(topY < bottomY) || !(leftX < rightX))
        {
            throw new InvalidOperationException(
                $"Recipient derived-crop line {lineIndex} quad is not ordered TL/TR/BR/BL");
        }
    }

    private static double RoundMilliseconds(Stopwatch stopwatch)
    {
        return Math.Round(stopwatch.Elapsed.TotalMilliseconds, 4, MidpointRounding.ToEven);
    }
}

internal sealed record RecipientDerivedCropShadowOptions(
    string Bundle,
    string PlanDirectory,
    string PlanSummarySha256,
    string Output)
{
    public static RecipientDerivedCropShadowOptions Parse(string[] args)
    {
        string? bundle = null;
        string? plan = null;
        string? planSummarySha256 = null;
        string? output = null;
        for (var index = 0; index < args.Length; index++)
        {
            var name = args[index];
            if (index + 1 >= args.Length)
            {
                throw new InvalidOperationException($"Missing value for recipient derived-crop argument {name}");
            }
            var value = args[++index];
            switch (name)
            {
                case "--bundle": bundle = SetOnce(bundle, value, name); break;
                case "--plan": plan = SetOnce(plan, value, name); break;
                case "--plan-summary-sha256": planSummarySha256 = SetOnce(planSummarySha256, value, name); break;
                case "--output": output = SetOnce(output, value, name); break;
                default: throw new InvalidOperationException($"Unknown recipient derived-crop argument: {name}");
            }
        }
        if (string.IsNullOrWhiteSpace(bundle)
            || string.IsNullOrWhiteSpace(plan)
            || string.IsNullOrWhiteSpace(planSummarySha256)
            || string.IsNullOrWhiteSpace(output))
        {
            throw new InvalidOperationException(
                "Usage: --bundle <verified-delivery> --plan <frozen-63-plan-directory> "
                + "--plan-summary-sha256 <lowercase-sha256> --output <fresh-directory>");
        }
        RecipientDerivedCropHash.RequireLowerSha256(
            planSummarySha256,
            "--plan-summary-sha256");
        return new RecipientDerivedCropShadowOptions(bundle, plan, planSummarySha256, output);
    }

    private static string SetOnce(string? current, string value, string name)
    {
        if (current is not null)
        {
            throw new InvalidOperationException($"Duplicate recipient derived-crop argument: {name}");
        }
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException($"Recipient derived-crop argument {name} must not be blank");
        }
        return value;
    }
}

internal static class RecipientDerivedCropPlanContract
{
    private static readonly HashSet<string> ImageExtensions = new(
        [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"],
        StringComparer.OrdinalIgnoreCase);
    private static readonly UTF8Encoding StrictUtf8 = new(
        encoderShouldEmitUTF8Identifier: false,
        throwOnInvalidBytes: true);

    public static RecipientDerivedCropPlanSelection Load(
        string planDirectory,
        string expectedSummarySha256)
    {
        RecipientDerivedCropHash.RequireLowerSha256(
            expectedSummarySha256,
            "--plan-summary-sha256");
        var directory = Path.GetFullPath(planDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (!Directory.Exists(directory))
        {
            throw new InvalidOperationException($"Derived-crop plan directory does not exist: {directory}");
        }
        RecipientDerivedCropFileContract.RequireRegularNonReparseDirectory(
            directory,
            "Derived-crop plan directory");
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            directory,
            "Derived-crop plan directory");
        var summaryPath = Path.Combine(directory, "summary.json");
        var summaryBytes = RecipientDerivedCropFileContract.ReadRegularFile(
            summaryPath,
            "Derived-crop plan summary");
        var summaryIdentity = RecipientDerivedCropFileIdentity.FromBytes(
            summaryPath,
            summaryBytes);
        if (!string.Equals(
            summaryIdentity.Sha256,
            expectedSummarySha256,
            StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Derived-crop plan summary SHA-256 differs: expected={expectedSummarySha256} "
                + $"actual={summaryIdentity.Sha256}");
        }

        using var summaryDocument = ParseJson(summaryBytes, "Derived-crop plan summary");
        var summary = summaryDocument.RootElement;
        RequirePlanSummary(summary);
        var filterContract = ReadExternalIdentity(
            RequireProperty(summary, "filter_contract", "plan summary"),
            "Derived-crop strict filter contract");
        var artifacts = RequireProperty(summary, "artifacts", "plan summary");
        var plansArtifact = ReadArtifact(
            directory,
            RequireProperty(artifacts, "plans", "plan summary artifacts"),
            "plans.jsonl",
            "Derived-crop plans artifact");
        var inputsArtifact = ReadArtifact(
            directory,
            RequireProperty(artifacts, "inputs", "plan summary artifacts"),
            "inputs.txt",
            "Derived-crop inputs artifact");
        var planRecords = ReadPlanRecords(plansArtifact.Bytes);
        var inputSources = ReadInputSources(inputsArtifact.Bytes);
        if (planRecords.Count != RecipientDerivedCropShadowProgram.ExpectedRecords
            || inputSources.Count != RecipientDerivedCropShadowProgram.ExpectedRecords)
        {
            throw new InvalidOperationException(
                "Derived-crop plan must contain exactly 63 plans and 63 input sources");
        }
        var comparer = SourceComparer;
        var seenSources = new HashSet<string>(comparer);
        var seenPlanIds = new HashSet<string>(StringComparer.Ordinal);
        for (var index = 0; index < planRecords.Count; index++)
        {
            var record = planRecords[index];
            if (!comparer.Equals(record.Source, inputSources[index]))
            {
                throw new InvalidOperationException(
                    $"Derived-crop inputs order differs from plans at index {index}");
            }
            if (!seenSources.Add(record.Source))
            {
                throw new InvalidOperationException($"Duplicate derived-crop source: {record.Source}");
            }
            if (!seenPlanIds.Add(record.PlanId))
            {
                throw new InvalidOperationException($"Duplicate derived-crop plan_id: {record.PlanId}");
            }
            VerifySourceIdentity(record, index);
        }

        return new RecipientDerivedCropPlanSelection(
            Directory: directory,
            Summary: summaryIdentity,
            Plans: plansArtifact.Identity,
            Inputs: inputsArtifact.Identity,
            FilterContract: filterContract,
            Records: planRecords.AsReadOnly());
    }

    public static void VerifyUnchanged(RecipientDerivedCropPlanSelection selection)
    {
        ArgumentNullException.ThrowIfNull(selection);
        VerifyIdentity(selection.Summary, "Derived-crop plan summary");
        VerifyIdentity(selection.Plans, "Derived-crop plans artifact");
        VerifyIdentity(selection.Inputs, "Derived-crop inputs artifact");
        VerifyIdentity(selection.FilterContract, "Derived-crop strict filter contract");
        for (var index = 0; index < selection.Records.Count; index++)
        {
            VerifySourceIdentity(selection.Records[index], index);
        }
    }

    private static void VerifyIdentity(
        RecipientDerivedCropFileIdentity expected,
        string description)
    {
        var actual = RecipientDerivedCropFileIdentity.FromFile(expected.Path, description);
        if (!actual.ContentEquals(expected))
        {
            throw new InvalidOperationException($"{description} changed while the diagnostic was running");
        }
    }

    private static void VerifySourceIdentity(RecipientDerivedCropPlanRecord record, int index)
    {
        RecipientDerivedCropFileContract.RequireRegularNonReparseFile(
            record.Source,
            $"Derived-crop source at index {index}");
        if (!ImageExtensions.Contains(Path.GetExtension(record.Source)))
        {
            throw new InvalidOperationException(
                $"Unsupported derived-crop image at index {index}: {record.Source}");
        }
        var current = RecipientDerivedCropFileIdentity.FromFile(
            record.Source,
            $"Derived-crop source at index {index}");
        if (!current.ContentEquals(record.SourceImage))
        {
            throw new InvalidOperationException(
                $"Derived-crop source identity differs from frozen plan: {record.Source}");
        }
    }

    private static RecipientDerivedCropArtifactSnapshot ReadArtifact(
        string directory,
        JsonElement element,
        string requiredPath,
        string description)
    {
        RequireObject(element, description);
        var relativePath = ReadRequiredString(element, "path", description);
        if (!string.Equals(relativePath, requiredPath, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{description} path must be exactly {requiredPath}");
        }
        var records = ReadRequiredInt(element, "records", description);
        if (records != RecipientDerivedCropShadowProgram.ExpectedRecords)
        {
            throw new InvalidOperationException($"{description} must bind exactly 63 records");
        }
        var expectedSha256 = ReadRequiredString(element, "sha256", description);
        RecipientDerivedCropHash.RequireLowerSha256(expectedSha256, $"{description} SHA-256");
        var expectedSize = ReadRequiredLong(element, "size_bytes", description);
        if (expectedSize <= 0)
        {
            throw new InvalidOperationException($"{description} size_bytes must be positive");
        }
        var path = RecipientDerivedCropFileContract.ResolveContainedFile(
            directory,
            relativePath,
            description);
        var bytes = RecipientDerivedCropFileContract.ReadRegularFile(path, description);
        var identity = RecipientDerivedCropFileIdentity.FromBytes(path, bytes);
        if (identity.SizeBytes != expectedSize
            || !string.Equals(identity.Sha256, expectedSha256, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{description} identity differs from plan summary");
        }
        return new RecipientDerivedCropArtifactSnapshot(identity, bytes);
    }

    private static RecipientDerivedCropFileIdentity ReadExternalIdentity(
        JsonElement element,
        string description)
    {
        RequireObject(element, description);
        var pathText = ReadRequiredString(element, "path", description);
        if (!Path.IsPathFullyQualified(pathText))
        {
            throw new InvalidOperationException($"{description} path must be absolute");
        }
        var path = Path.GetFullPath(pathText);
        var expectedSha256 = ReadRequiredString(element, "sha256", description);
        RecipientDerivedCropHash.RequireLowerSha256(expectedSha256, $"{description} SHA-256");
        var expectedSize = ReadRequiredLong(element, "size_bytes", description);
        if (expectedSize <= 0)
        {
            throw new InvalidOperationException($"{description} must be non-empty");
        }
        var identity = RecipientDerivedCropFileIdentity.FromFile(path, description);
        if (identity.SizeBytes != expectedSize
            || !string.Equals(identity.Sha256, expectedSha256, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{description} identity differs from plan summary");
        }
        return identity;
    }

    private static void RequirePlanSummary(JsonElement summary)
    {
        RequireObject(summary, "Derived-crop plan summary");
        if (ReadRequiredInt(summary, "schema_version", "plan summary") != 1
            || !string.Equals(
                ReadRequiredString(summary, "kind", "plan summary"),
                RecipientDerivedCropShadowProgram.PlanSummaryKind,
                StringComparison.Ordinal)
            || !ReadRequiredBool(summary, "diagnostic_only", "plan summary")
            || ReadRequiredBool(summary, "formal_delivery_gate", "plan summary")
            || ReadRequiredBool(summary, "candidate_write_enabled", "plan summary")
            || ReadRequiredBool(summary, "ocr_rerun", "plan summary")
            || ReadRequiredBool(summary, "production_output_changed", "plan summary")
            || ReadRequiredInt(summary, "records", "plan summary")
                != RecipientDerivedCropShadowProgram.ExpectedRecords)
        {
            throw new InvalidOperationException(
                "Derived-crop plan summary violates the diagnostic-only 63-record contract");
        }
        var frozen = RequireProperty(summary, "frozen_v4", "plan summary");
        if (ReadRequiredInt(frozen, "formal_failures", "frozen_v4") != 204
            || ReadRequiredInt(frozen, "candidate_records", "frozen_v4") != 75
            || ReadRequiredInt(frozen, "remaining_records", "frozen_v4") != 129
            || ReadRequiredInt(frozen, "remaining_with_global_gate_failures", "frozen_v4") != 66
            || ReadRequiredInt(frozen, "remaining_with_clear_global_gates", "frozen_v4") != 63)
        {
            throw new InvalidOperationException("Derived-crop plan is not the frozen v4 75/129 and 66/63 cohort");
        }
        var cropNames = RequireProperty(summary, "crop_names", "plan summary");
        var cropNameValues = cropNames.ValueKind == JsonValueKind.Array
            ? cropNames.EnumerateArray().ToArray()
            : Array.Empty<JsonElement>();
        if (cropNames.ValueKind != JsonValueKind.Array
            || cropNames.GetArrayLength() != 2
            || cropNameValues[0].ValueKind != JsonValueKind.String
            || cropNameValues[1].ValueKind != JsonValueKind.String
            || !string.Equals(cropNameValues[0].GetString(), RecipientDerivedCropShadowProgram.Crop4, StringComparison.Ordinal)
            || !string.Equals(cropNameValues[1].GetString(), RecipientDerivedCropShadowProgram.Crop5, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Derived-crop plan crop_names differ from crop4/crop5 contract");
        }
        var route = RequireProperty(summary, "route_contract", "plan summary");
        if (!ReadRequiredBool(route, "crop4_requires_exact_match_with_existing_strict_crop", "route_contract")
            || !ReadRequiredBool(route, "crop5_requires_unique_exact_crop4_crop5_agreement", "route_contract")
            || ReadRequiredDouble(route, "minimum_line_confidence", "route_contract") != 0.80
            || ReadRequiredDouble(route, "minimum_recipient_detector_score", "route_contract") != 0.68
            || !ReadRequiredBool(route, "requires_ordinary_25pct_geometry", "route_contract")
            || !ReadRequiredBool(route, "requires_alternative_envelope", "route_contract")
            || ReadRequiredBool(route, "candidate_write_enabled", "route_contract"))
        {
            throw new InvalidOperationException("Derived-crop route contract differs from frozen protection floors");
        }
        var producer = RequireProperty(summary, "required_layout_producer", "plan summary");
        if (!string.Equals(
                ReadRequiredString(producer, "api", "required_layout_producer"),
                "PaddleOcrEngine.RecognizeLayoutDiagnostic",
                StringComparison.Ordinal)
            || !string.Equals(
                ReadRequiredString(producer, "execution_provider", "required_layout_producer"),
                "cpu",
                StringComparison.Ordinal)
            || !string.Equals(
                ReadRequiredString(producer, "rectification", "required_layout_producer"),
                RecipientDerivedCropShadowProgram.PlanRectification,
                StringComparison.Ordinal)
            || !ReadRequiredBool(
                producer,
                "requires_raw_quad_crop_and_rectified_coordinates",
                "required_layout_producer")
            || !ReadRequiredBool(
                producer,
                "requires_verified_paddle_bundle_identity",
                "required_layout_producer")
            || !string.Equals(
                ReadRequiredString(producer, "required_summary_kind", "required_layout_producer"),
                RecipientDerivedCropShadowProgram.SummaryKind,
                StringComparison.Ordinal)
            || !string.Equals(
                ReadRequiredString(producer, "required_record_kind", "required_layout_producer"),
                RecipientDerivedCropShadowProgram.RecordKind,
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Derived-crop plan requires a different layout producer contract");
        }
    }

    private static List<RecipientDerivedCropPlanRecord> ReadPlanRecords(byte[] bytes)
    {
        var text = StrictUtf8.GetString(bytes);
        if (text.Length > 0 && text[0] == '\uFEFF')
        {
            throw new InvalidOperationException("Derived-crop plans artifact must be UTF-8 without BOM");
        }
        var records = new List<RecipientDerivedCropPlanRecord>(
            RecipientDerivedCropShadowProgram.ExpectedRecords);
        using var reader = new StringReader(text);
        var lineNumber = 0;
        while (reader.ReadLine() is { } line)
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line)
                || !string.Equals(line, line.Trim(), StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Derived-crop plans artifact contains a blank line at {lineNumber}");
            }
            using var document = ParseJson(
                Encoding.UTF8.GetBytes(line),
                $"Derived-crop plan line {lineNumber}");
            records.Add(ParsePlanRecord(document.RootElement, lineNumber - 1));
        }
        return records;
    }

    private static List<string> ReadInputSources(byte[] bytes)
    {
        var text = StrictUtf8.GetString(bytes);
        if (text.Length > 0 && text[0] == '\uFEFF')
        {
            throw new InvalidOperationException("Derived-crop inputs artifact must be UTF-8 without BOM");
        }
        var sources = new List<string>(RecipientDerivedCropShadowProgram.ExpectedRecords);
        using var reader = new StringReader(text);
        var lineNumber = 0;
        while (reader.ReadLine() is { } rawLine)
        {
            lineNumber++;
            if (rawLine.Length == 0
                || !string.Equals(rawLine, rawLine.Trim(), StringComparison.Ordinal)
                || !Path.IsPathFullyQualified(rawLine))
            {
                throw new InvalidOperationException(
                    $"Invalid absolute derived-crop input source at line {lineNumber}");
            }
            sources.Add(Path.GetFullPath(rawLine));
        }
        return sources;
    }

    private static RecipientDerivedCropPlanRecord ParsePlanRecord(JsonElement root, int index)
    {
        RequireObject(root, $"Derived-crop plan {index}");
        foreach (var forbidden in new[] { "candidate", "shadow_candidate", "delivery_value", "fields" })
        {
            if (root.TryGetProperty(forbidden, out _))
            {
                throw new InvalidOperationException(
                    $"Derived-crop plan {index} contains forbidden production field {forbidden}");
            }
        }
        if (ReadRequiredInt(root, "schema_version", $"plan {index}") != 1
            || !string.Equals(
                ReadRequiredString(root, "kind", $"plan {index}"),
                RecipientDerivedCropShadowProgram.PlanRecordKind,
                StringComparison.Ordinal)
            || !ReadRequiredBool(root, "diagnostic_only", $"plan {index}")
            || ReadRequiredBool(root, "formal_delivery_gate", $"plan {index}")
            || ReadRequiredBool(root, "candidate_write_enabled", $"plan {index}"))
        {
            throw new InvalidOperationException($"Derived-crop plan {index} is not diagnostic-only");
        }
        var sourceText = ReadRequiredString(root, "source", $"plan {index}");
        if (!Path.IsPathFullyQualified(sourceText))
        {
            throw new InvalidOperationException($"Derived-crop plan {index} source must be absolute");
        }
        var source = Path.GetFullPath(sourceText);
        var sourceImageElement = RequireProperty(root, "source_image", $"plan {index}");
        var sourceImagePathText = ReadRequiredString(
            sourceImageElement,
            "path",
            $"plan {index} source_image");
        if (!Path.IsPathFullyQualified(sourceImagePathText))
        {
            throw new InvalidOperationException(
                $"Derived-crop plan {index} source_image path must be absolute");
        }
        var sourceImagePath = Path.GetFullPath(sourceImagePathText);
        if (!SourceComparer.Equals(source, sourceImagePath))
        {
            throw new InvalidOperationException($"Derived-crop plan {index} source_image path differs from source");
        }
        var sourceSha256 = ReadRequiredString(
            sourceImageElement,
            "sha256",
            $"plan {index} source_image");
        RecipientDerivedCropHash.RequireLowerSha256(
            sourceSha256,
            $"plan {index} source image SHA-256");
        var sourceSize = ReadRequiredLong(
            sourceImageElement,
            "size_bytes",
            $"plan {index} source_image");
        if (sourceSize <= 0)
        {
            throw new InvalidOperationException($"Derived-crop plan {index} source image must be non-empty");
        }
        if (!string.Equals(
            ReadRequiredString(root, "rectification", $"plan {index}"),
            RecipientDerivedCropShadowProgram.PlanRectification,
            StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Derived-crop plan {index} rectification is not max_side_1600");
        }
        var sizeElement = RequireProperty(root, "rectified_size", $"plan {index}");
        var size = new RecipientDerivedCropSize(
            ReadRequiredInt(sizeElement, "width", $"plan {index} rectified_size"),
            ReadRequiredInt(sizeElement, "height", $"plan {index} rectified_size"));
        if (size.Width < 2 || size.Height < 2 || Math.Max(size.Width, size.Height) > 1600)
        {
            throw new InvalidOperationException($"Derived-crop plan {index} rectified size violates max-side-1600");
        }
        RequireGlobalGates(root, index);
        RequireExistingAttempts(root, index);
        var geometry = ReadGeometry(root, size, index);
        var expectedCrops = DeriveCrops(geometry, size);
        var crops = ReadCrops(root, size, index);
        if (!crops[0].ValueEquals(expectedCrops[0]) || !crops[1].ValueEquals(expectedCrops[1]))
        {
            throw new InvalidOperationException($"Derived-crop plan {index} crop geometry is not canonical");
        }
        var planId = ReadRequiredString(root, "plan_id", $"plan {index}");
        RecipientDerivedCropHash.RequireLowerSha256(planId, $"plan {index} plan_id");
        var canonicalPlanId = CanonicalPlanId(root);
        if (!string.Equals(planId, canonicalPlanId, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Derived-crop plan {index} plan_id differs from canonical plan payload");
        }
        return new RecipientDerivedCropPlanRecord(
            Index: index,
            Source: source,
            SourceImage: new RecipientDerivedCropFileIdentity(source, sourceSha256, sourceSize),
            RectifiedSize: size,
            Crops: crops.AsReadOnly(),
            PlanId: planId);
    }

    private static void RequireGlobalGates(JsonElement root, int index)
    {
        var gates = RequireProperty(root, "global_gate_evidence", $"plan {index}");
        var failures = RequireProperty(gates, "global_gate_failures", $"plan {index} gates");
        var score = ReadRequiredDouble(gates, "recipient_detector_score", $"plan {index} gates");
        if (failures.ValueKind != JsonValueKind.Array
            || failures.GetArrayLength() != 0
            || !ReadRequiredBool(gates, "ordinary_25pct_geometry_verified", $"plan {index} gates")
            || !ReadRequiredBool(gates, "alternative_envelope_verified", $"plan {index} gates")
            || ReadRequiredDouble(gates, "minimum_recipient_detector_score", $"plan {index} gates") != 0.68
            || !double.IsFinite(score)
            || score < 0.68
            || score > 1.0)
        {
            throw new InvalidOperationException($"Derived-crop plan {index} does not preserve every global gate");
        }
    }

    private static void RequireExistingAttempts(JsonElement root, int index)
    {
        var attempts = RequireProperty(root, "existing_attempts", $"plan {index}");
        RequireObject(attempts, $"plan {index} existing_attempts");
        var properties = attempts.EnumerateObject().ToArray();
        var expectedNames = new HashSet<string>(
            ["first", "retry", "right_value"],
            StringComparer.Ordinal);
        if (properties.Length != expectedNames.Count
            || !properties.Select(property => property.Name).ToHashSet(StringComparer.Ordinal).SetEquals(expectedNames))
        {
            throw new InvalidOperationException(
                $"Derived-crop plan {index} existing_attempts must contain exactly first/retry/right_value");
        }
        foreach (var property in properties)
        {
            var attempt = property.Value;
            RequireObject(attempt, $"plan {index} attempt {property.Name}");
            var lines = RequireProperty(
                attempt,
                "lines",
                $"plan {index} attempt {property.Name}");
            if (lines.ValueKind != JsonValueKind.Array)
            {
                throw new InvalidOperationException(
                    $"Derived-crop plan {index} attempt {property.Name} lines must be an array");
            }
            var lineIndex = 0;
            foreach (var line in lines.EnumerateArray())
            {
                RequireObject(line, $"plan {index} attempt {property.Name} line {lineIndex}");
                if (ReadRequiredInt(
                        line,
                        "index",
                        $"plan {index} attempt {property.Name} line {lineIndex}") != lineIndex)
                {
                    throw new InvalidOperationException(
                        $"Derived-crop plan {index} attempt {property.Name} line indices are not contiguous");
                }
                var text = RequireProperty(
                    line,
                    "text",
                    $"plan {index} attempt {property.Name} line {lineIndex}");
                if (text.ValueKind != JsonValueKind.String)
                {
                    throw new InvalidOperationException(
                        $"Derived-crop plan {index} attempt {property.Name} line text must be a string");
                }
                var confidence = ReadRequiredDouble(
                    line,
                    "confidence",
                    $"plan {index} attempt {property.Name} line {lineIndex}");
                if (confidence < 0.0 || confidence > 1.0)
                {
                    throw new InvalidOperationException(
                        $"Derived-crop plan {index} attempt {property.Name} line confidence is outside [0, 1]");
                }
                lineIndex++;
            }
        }
    }

    internal static string CanonicalPlanId(JsonElement root)
    {
        RequireObject(root, "Derived-crop canonical plan payload");
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions
        {
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            Indented = false,
            SkipValidation = false,
        }))
        {
            WriteCanonicalJson(writer, root, omitRootPlanId: true);
        }
        return RecipientDerivedCropHash.Sha256(buffer.WrittenSpan.ToArray());
    }

    private static void WriteCanonicalJson(
        Utf8JsonWriter writer,
        JsonElement element,
        bool omitRootPlanId = false)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (var property in element.EnumerateObject()
                    .Where(property => !(omitRootPlanId
                        && string.Equals(property.Name, "plan_id", StringComparison.Ordinal)))
                    .OrderBy(property => property.Name, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(property.Name);
                    WriteCanonicalJson(writer, property.Value);
                }
                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (var item in element.EnumerateArray())
                {
                    WriteCanonicalJson(writer, item);
                }
                writer.WriteEndArray();
                break;
            case JsonValueKind.String:
                writer.WriteStringValue(element.GetString());
                break;
            case JsonValueKind.Number:
                writer.WriteRawValue(PythonCanonicalNumber(element), skipInputValidation: false);
                break;
            case JsonValueKind.True:
                writer.WriteBooleanValue(true);
                break;
            case JsonValueKind.False:
                writer.WriteBooleanValue(false);
                break;
            case JsonValueKind.Null:
                writer.WriteNullValue();
                break;
            default:
                throw new InvalidOperationException(
                    $"Unsupported JSON value in canonical plan payload: {element.ValueKind}");
        }
    }

    private static string PythonCanonicalNumber(JsonElement element)
    {
        var raw = element.GetRawText();
        if (!raw.Contains('.')
            && !raw.Contains('e')
            && !raw.Contains('E')
            && element.TryGetInt64(out var integer))
        {
            return integer.ToString(CultureInfo.InvariantCulture);
        }
        if (!element.TryGetDouble(out var value) || !double.IsFinite(value))
        {
            throw new InvalidOperationException("Canonical plan number must be finite");
        }
        var rendered = value.ToString("R", CultureInfo.InvariantCulture).ToLowerInvariant();
        var exponentIndex = rendered.IndexOf('e');
        if (exponentIndex >= 0)
        {
            var mantissa = rendered[..exponentIndex];
            var exponent = int.Parse(rendered[(exponentIndex + 1)..], CultureInfo.InvariantCulture);
            var magnitude = Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture);
            var exponentText = exponent >= 0 ? $"+{magnitude}" : $"-{magnitude}";
            return $"{mantissa}e{exponentText}";
        }
        return rendered.Contains('.') ? rendered : rendered + ".0";
    }

    private static RecipientDerivedCropGeometry ReadGeometry(
        JsonElement root,
        RecipientDerivedCropSize size,
        int index)
    {
        var geometry = RequireProperty(root, "detector_geometry", $"plan {index}");
        return new RecipientDerivedCropGeometry(
            ReadDoubleBox(geometry, "amount_box", size, $"plan {index} amount_box"),
            ReadDoubleBox(geometry, "recipient_box", size, $"plan {index} recipient_box"),
            ReadDoubleBox(geometry, "payment_box", size, $"plan {index} payment_box"));
    }

    private static double[] ReadDoubleBox(
        JsonElement parent,
        string property,
        RecipientDerivedCropSize size,
        string description)
    {
        var element = RequireProperty(parent, property, description);
        if (element.ValueKind != JsonValueKind.Array || element.GetArrayLength() != 4)
        {
            throw new InvalidOperationException($"{description} must contain four coordinates");
        }
        var values = element.EnumerateArray().Select(value => ReadFiniteDouble(value, description)).ToArray();
        if (values[2] <= values[0]
            || values[3] <= values[1]
            || values[0] < -1.0
            || values[1] < -1.0
            || values[2] > size.Width + 1.0
            || values[3] > size.Height + 1.0)
        {
            throw new InvalidOperationException($"{description} is invalid or outside rectified bounds");
        }
        return values;
    }

    private static List<RecipientDerivedCropPlanCrop> ReadCrops(
        JsonElement root,
        RecipientDerivedCropSize size,
        int index)
    {
        var element = RequireProperty(root, "crops", $"plan {index}");
        if (element.ValueKind != JsonValueKind.Array || element.GetArrayLength() != 2)
        {
            throw new InvalidOperationException($"Derived-crop plan {index} must contain exactly crop4 and crop5");
        }
        var crops = new List<RecipientDerivedCropPlanCrop>(2);
        var expectedNames = new[]
        {
            RecipientDerivedCropShadowProgram.Crop4,
            RecipientDerivedCropShadowProgram.Crop5,
        };
        var cropIndex = 0;
        foreach (var crop in element.EnumerateArray())
        {
            var description = $"plan {index} crop {cropIndex}";
            var name = ReadRequiredString(crop, "name", description);
            if (!string.Equals(name, expectedNames[cropIndex], StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"{description} name differs from frozen route");
            }
            var boxElement = RequireProperty(crop, "rectified_box", description);
            if (boxElement.ValueKind != JsonValueKind.Array || boxElement.GetArrayLength() != 4)
            {
                throw new InvalidOperationException($"{description} rectified_box must contain four integers");
            }
            var coordinates = boxElement.EnumerateArray()
                .Select(value => value.TryGetInt32(out var parsed)
                    ? parsed
                    : throw new InvalidOperationException($"{description} rectified_box must contain integers"))
                .ToArray();
            var box = new RecipientDerivedCropBox(
                coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
            if (box.Left < 0
                || box.Top < 0
                || box.Right > size.Width
                || box.Bottom > size.Height
                || box.Width <= 0
                || box.Height <= 0
                || ReadRequiredInt(crop, "width", description) != box.Width
                || ReadRequiredInt(crop, "height", description) != box.Height
                || !string.Equals(
                    ReadRequiredString(crop, "pixel_box_semantics", description),
                    "left_top_inclusive_right_bottom_exclusive",
                    StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"{description} dimensions or bounds are invalid");
            }
            crops.Add(new RecipientDerivedCropPlanCrop(name, box));
            cropIndex++;
        }
        return crops;
    }

    internal static IReadOnlyList<RecipientDerivedCropPlanCrop> DeriveCrops(
        RecipientDerivedCropGeometry geometry,
        RecipientDerivedCropSize size)
    {
        var amount = geometry.Amount;
        var recipient = geometry.Recipient;
        var payment = geometry.Payment;
        var recipientWidth = recipient[2] - recipient[0];
        var recipientHeight = recipient[3] - recipient[1];
        var amountCenterY = (amount[1] + amount[3]) * 0.5;
        var recipientCenterX = (recipient[0] + recipient[2]) * 0.5;
        var recipientCenterY = (recipient[1] + recipient[3]) * 0.5;
        var paymentCenterY = (payment[1] + payment[3]) * 0.5;
        if (!(amountCenterY < recipientCenterY && recipientCenterY < paymentCenterY))
        {
            throw new InvalidOperationException("Derived crops require amount < recipient < payment centers");
        }
        var upperMidpoint = (amountCenterY + recipientCenterY) * 0.5;
        var lowerMidpoint = (recipientCenterY + paymentCenterY) * 0.5;
        var right = Math.Min(size.Width, recipient[2] + 0.08 * recipientWidth);

        RecipientDerivedCropPlanCrop Rectangle(
            string name,
            double left,
            double top,
            double bottom)
        {
            var box = new RecipientDerivedCropBox(
                Math.Max(0, (int)Math.Floor(left)),
                Math.Max(0, (int)Math.Floor(top)),
                Math.Min(size.Width, (int)Math.Ceiling(right)),
                Math.Min(size.Height, (int)Math.Ceiling(bottom)));
            if (box.Width <= 0
                || box.Height <= 0
                || !(box.Left <= recipientCenterX && recipientCenterX < box.Right)
                || !(box.Top <= recipientCenterY && recipientCenterY < box.Bottom)
                || (box.Top <= amountCenterY && amountCenterY < box.Bottom)
                || (box.Top <= paymentCenterY && paymentCenterY < box.Bottom))
            {
                throw new InvalidOperationException($"Canonical derived {name} geometry is invalid");
            }
            return new RecipientDerivedCropPlanCrop(name, box);
        }

        var crop4 = Rectangle(
            RecipientDerivedCropShadowProgram.Crop4,
            Math.Max(size.Width * 0.28, recipient[0] + 0.20 * recipientWidth),
            Math.Max(0.0, Math.Max(upperMidpoint, recipient[1] - 0.35 * recipientHeight)),
            Math.Min(size.Height, Math.Min(lowerMidpoint, recipient[3] + 0.35 * recipientHeight)));
        var crop5 = Rectangle(
            RecipientDerivedCropShadowProgram.Crop5,
            Math.Max(size.Width * 0.36, recipient[0] + 0.32 * recipientWidth),
            Math.Max(0.0, Math.Max(upperMidpoint, recipient[1] - 0.08 * recipientHeight)),
            Math.Min(size.Height, Math.Min(lowerMidpoint, recipient[3] + 0.08 * recipientHeight)));
        var minimumLeftDelta = Math.Max(2, (int)Math.Floor(size.Width * 0.04));
        if (crop5.Box.Left - crop4.Box.Left < minimumLeftDelta
            || crop4.Box.ValueEquals(crop5.Box))
        {
            throw new InvalidOperationException(
                "Canonical crop4/crop5 do not have independent horizontal context");
        }
        return Array.AsReadOnly(new[] { crop4, crop5 });
    }

    private static JsonDocument ParseJson(byte[] bytes, string description)
    {
        try
        {
            var document = JsonDocument.Parse(bytes, new JsonDocumentOptions
            {
                AllowTrailingCommas = false,
                CommentHandling = JsonCommentHandling.Disallow,
            });
            RequireNoDuplicateProperties(document.RootElement, description);
            return document;
        }
        catch (JsonException error)
        {
            throw new InvalidOperationException($"Invalid {description}: {error.Message}", error);
        }
    }

    private static void RequireNoDuplicateProperties(JsonElement element, string description)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in element.EnumerateObject())
            {
                if (!seen.Add(property.Name))
                {
                    throw new InvalidOperationException(
                        $"{description} contains duplicate property {property.Name}");
                }
                RequireNoDuplicateProperties(property.Value, description);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                RequireNoDuplicateProperties(item, description);
            }
        }
    }

    private static JsonElement RequireProperty(
        JsonElement parent,
        string property,
        string description)
    {
        RequireObject(parent, description);
        if (!parent.TryGetProperty(property, out var value))
        {
            throw new InvalidOperationException($"{description} lacks required property {property}");
        }
        return value;
    }

    private static void RequireObject(JsonElement element, string description)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidOperationException($"{description} must be a JSON object");
        }
    }

    private static string ReadRequiredString(
        JsonElement parent,
        string property,
        string description)
    {
        var element = RequireProperty(parent, property, description);
        if (element.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(element.GetString()))
        {
            throw new InvalidOperationException($"{description}.{property} must be a non-empty string");
        }
        return element.GetString()!;
    }

    private static int ReadRequiredInt(JsonElement parent, string property, string description)
    {
        var element = RequireProperty(parent, property, description);
        if (!element.TryGetInt32(out var value))
        {
            throw new InvalidOperationException($"{description}.{property} must be an integer");
        }
        return value;
    }

    private static long ReadRequiredLong(JsonElement parent, string property, string description)
    {
        var element = RequireProperty(parent, property, description);
        if (!element.TryGetInt64(out var value))
        {
            throw new InvalidOperationException($"{description}.{property} must be an integer");
        }
        return value;
    }

    private static bool ReadRequiredBool(JsonElement parent, string property, string description)
    {
        var element = RequireProperty(parent, property, description);
        if (element.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw new InvalidOperationException($"{description}.{property} must be a boolean");
        }
        return element.GetBoolean();
    }

    private static double ReadRequiredDouble(JsonElement parent, string property, string description)
    {
        return ReadFiniteDouble(RequireProperty(parent, property, description), $"{description}.{property}");
    }

    private static double ReadFiniteDouble(JsonElement element, string description)
    {
        if (element.ValueKind != JsonValueKind.Number
            || !element.TryGetDouble(out var value)
            || !double.IsFinite(value))
        {
            throw new InvalidOperationException($"{description} must be finite");
        }
        return value;
    }

    private static StringComparer SourceComparer => OperatingSystem.IsWindows()
        ? StringComparer.OrdinalIgnoreCase
        : StringComparer.Ordinal;
}

internal static class RecipientDerivedCropOutputContract
{
    public static RecipientDerivedCropOutput ResolveFreshOutput(string output)
    {
        var fullPath = Path.GetFullPath(output)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var name = Path.GetFileName(fullPath);
        var parent = Path.GetDirectoryName(fullPath);
        if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(parent))
        {
            throw new InvalidOperationException(
                "Recipient derived-crop output must be a non-root directory path");
        }
        if (File.Exists(fullPath) || Directory.Exists(fullPath))
        {
            throw new InvalidOperationException(
                $"Refusing to overwrite recipient derived-crop output: {fullPath}");
        }
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            parent,
            "Recipient derived-crop output parent");
        return new RecipientDerivedCropOutput(fullPath, parent, name);
    }

    public static void RequireDisjoint(
        RecipientDerivedCropOutput output,
        string protectedDirectory,
        string description)
    {
        var protectedPath = Path.GetFullPath(protectedDirectory);
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            output.Parent,
            "Recipient derived-crop output parent");
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            protectedPath,
            description);
        if (IsWithin(output.FullPath, protectedPath)
            || IsWithin(protectedPath, output.FullPath))
        {
            throw new InvalidOperationException(
                $"Recipient derived-crop output must be disjoint from {description}");
        }
    }

    public static void VerifyPublicationParent(RecipientDerivedCropOutput output)
    {
        ArgumentNullException.ThrowIfNull(output);
        if (!Directory.Exists(output.Parent))
        {
            throw new InvalidOperationException(
                $"Recipient derived-crop output parent does not exist: {output.Parent}");
        }
        RecipientDerivedCropFileContract.RequireRegularNonReparseDirectory(
            output.Parent,
            "Recipient derived-crop output parent");
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            output.Parent,
            "Recipient derived-crop output parent");
        if (File.Exists(output.FullPath) || Directory.Exists(output.FullPath))
        {
            throw new InvalidOperationException(
                $"Refusing to overwrite recipient derived-crop output: {output.FullPath}");
        }
    }

    public static void VerifyOwnedStage(
        RecipientDerivedCropOutput output,
        string stage)
    {
        VerifyPublicationParent(output);
        RecipientDerivedCropFileContract.RequireRegularNonReparseDirectory(
            stage,
            "Recipient derived-crop owned stage");
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            stage,
            "Recipient derived-crop owned stage");
        var stageParent = Directory.GetParent(Path.GetFullPath(stage))?.FullName;
        var comparer = OperatingSystem.IsWindows()
            ? StringComparer.OrdinalIgnoreCase
            : StringComparer.Ordinal;
        if (stageParent is null || !comparer.Equals(stageParent, output.Parent))
        {
            throw new InvalidOperationException(
                "Recipient derived-crop owned stage moved outside its verified output parent");
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
            // Preserve the inference/publication error.
        }
    }

    private static bool IsWithin(string candidate, string root)
    {
        var relative = Path.GetRelativePath(root, candidate);
        return !Path.IsPathRooted(relative)
            && !string.Equals(relative, "..", StringComparison.Ordinal)
            && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            && !relative.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal);
    }
}

internal static class RecipientDerivedCropFileContract
{
    public static void RequireNoReparseDirectoryChain(string path, string description)
    {
        DirectoryInfo? current = new(Path.GetFullPath(path));
        while (current is not null)
        {
            if (current.Exists
                && (current.Attributes & FileAttributes.ReparsePoint) != 0)
            {
                throw new InvalidOperationException(
                    $"{description} contains a reparse/junction directory: {current.FullName}");
            }
            current = current.Parent;
        }
    }

    public static void RequireRegularNonReparseDirectory(string path, string description)
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

    public static byte[] ReadRegularFile(string path, string description)
    {
        if (!File.Exists(path))
        {
            throw new InvalidOperationException($"{description} does not exist: {path}");
        }
        RequireRegularNonReparseFile(path, description);
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read);
        if (stream.Length > int.MaxValue)
        {
            throw new InvalidOperationException($"{description} is too large to read atomically: {path}");
        }
        var bytes = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytes);
        return bytes;
    }

    public static string ResolveContainedFile(
        string directory,
        string relativePath,
        string description)
    {
        if (Path.IsPathFullyQualified(relativePath))
        {
            throw new InvalidOperationException($"{description} path must be relative");
        }
        var path = Path.GetFullPath(Path.Combine(directory, relativePath));
        var relative = Path.GetRelativePath(directory, path);
        if (Path.IsPathRooted(relative)
            || string.Equals(relative, "..", StringComparison.Ordinal)
            || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || relative.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{description} escapes its directory");
        }
        return path;
    }
}

internal static class RecipientDerivedCropHash
{
    public static string Sha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    public static void RequireLowerSha256(string value, string description)
    {
        if (value.Length != 64
            || value.Any(character => !IsLowerHex(character)))
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

internal static class RecipientDerivedCropJson
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

internal sealed record RecipientDerivedCropArtifactSnapshot(
    RecipientDerivedCropFileIdentity Identity,
    byte[] Bytes);

internal sealed record RecipientDerivedCropPlanSelection(
    string Directory,
    RecipientDerivedCropFileIdentity Summary,
    RecipientDerivedCropFileIdentity Plans,
    RecipientDerivedCropFileIdentity Inputs,
    RecipientDerivedCropFileIdentity FilterContract,
    IReadOnlyList<RecipientDerivedCropPlanRecord> Records);

internal sealed record RecipientDerivedCropPlanRecord(
    int Index,
    string Source,
    RecipientDerivedCropFileIdentity SourceImage,
    RecipientDerivedCropSize RectifiedSize,
    IReadOnlyList<RecipientDerivedCropPlanCrop> Crops,
    string PlanId);

internal sealed record RecipientDerivedCropGeometry(
    double[] Amount,
    double[] Recipient,
    double[] Payment);

internal sealed record RecipientDerivedCropPlanCrop(
    string Name,
    RecipientDerivedCropBox Box)
{
    public bool ValueEquals(RecipientDerivedCropPlanCrop other)
    {
        return string.Equals(Name, other.Name, StringComparison.Ordinal)
            && Box.ValueEquals(other.Box);
    }
}

internal sealed record RecipientDerivedCropBox(int Left, int Top, int Right, int Bottom)
{
    public int Width => Right - Left;
    public int Height => Bottom - Top;
    public int[] ToArray() => [Left, Top, Right, Bottom];

    public bool ValueEquals(RecipientDerivedCropBox other)
    {
        return Left == other.Left
            && Top == other.Top
            && Right == other.Right
            && Bottom == other.Bottom;
    }
}

internal sealed record RecipientDerivedCropSize(int Width, int Height);

internal sealed record RecipientDerivedCropFileIdentity(
    string Path,
    string Sha256,
    long SizeBytes)
{
    public static RecipientDerivedCropFileIdentity FromBytes(string path, byte[] bytes)
    {
        return new RecipientDerivedCropFileIdentity(
            System.IO.Path.GetFullPath(path),
            RecipientDerivedCropHash.Sha256(bytes),
            bytes.LongLength);
    }

    public static RecipientDerivedCropFileIdentity FromFile(string path, string description)
    {
        return FromBytes(path, RecipientDerivedCropFileContract.ReadRegularFile(path, description));
    }

    public bool ContentEquals(RecipientDerivedCropFileIdentity other)
    {
        var comparer = OperatingSystem.IsWindows()
            ? StringComparer.OrdinalIgnoreCase
            : StringComparer.Ordinal;
        return comparer.Equals(Path, other.Path)
            && string.Equals(Sha256, other.Sha256, StringComparison.Ordinal)
            && SizeBytes == other.SizeBytes;
    }
}

internal sealed record RecipientDerivedCropOutput(string FullPath, string Parent, string Name);

internal sealed record RecipientDerivedCropLine(
    int Index,
    string Text,
    float Confidence,
    bool PassesDropScore,
    float[][] QuadCrop,
    float[][] QuadRectified);

internal sealed record RecipientDerivedCropLayout(
    string Name,
    int[] RectifiedBox,
    int Width,
    int Height,
    IReadOnlyList<RecipientDerivedCropLine> Lines);

internal sealed record RecipientDerivedCropTiming(
    double ImageLoad,
    double Rectification,
    double Crop4LayoutOcr,
    double Crop5LayoutOcr,
    double Total);

internal sealed record RecipientDerivedCropRecord(
    int SchemaVersion,
    string Kind,
    bool DiagnosticOnly,
    bool FormalDeliveryGate,
    bool CandidateWriteEnabled,
    bool ProductionOutputChanged,
    int Index,
    string Source,
    string SourceImageSha256,
    long SourceImageSizeBytes,
    string ExecutionProvider,
    string Rectification,
    RecipientDerivedCropSize RectifiedSize,
    string PlanId,
    string QuadCoordinateSpace,
    string ConfidenceSemantics,
    IReadOnlyList<RecipientDerivedCropLayout> Crops,
    RecipientDerivedCropTiming TimingMs);

internal sealed record RecipientDerivedCropFileEvidence(
    string Path,
    string Sha256,
    long SizeBytes);

internal static class RecipientDerivedCropBundleSnapshot
{
    public static void Create(
        PaddleOcrDeliveryBundle source,
        string destinationDirectory)
    {
        ArgumentNullException.ThrowIfNull(source);
        var destination = Path.GetFullPath(destinationDirectory);
        if (File.Exists(destination) || Directory.Exists(destination))
        {
            throw new InvalidOperationException(
                $"Refusing to overwrite private Paddle OCR bundle snapshot: {destination}");
        }
        Directory.CreateDirectory(destination);
        try
        {
            RecipientDerivedCropFileContract.RequireRegularNonReparseDirectory(
                destination,
                "Private Paddle OCR bundle snapshot");
            RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
                destination,
                "Private Paddle OCR bundle snapshot");
            var written = new HashSet<string>(
                OperatingSystem.IsWindows()
                    ? StringComparer.OrdinalIgnoreCase
                    : StringComparer.Ordinal);

            var contractBytes = RecipientDerivedCropFileContract.ReadRegularFile(
                source.ContractPath,
                "Source Paddle OCR delivery contract snapshot");
            VerifyBoundBytes(
                contractBytes,
                source.ContractSha256,
                expectedSizeBytes: null,
                "Source Paddle OCR delivery contract");
            WriteSnapshotFile(
                destination,
                PaddleOcrDeliveryBundle.ContractFileName,
                contractBytes,
                written,
                "Paddle OCR delivery contract");
            CopyVerifiedComponent(destination, source.DetModel.File, written, "detector");
            CopyVerifiedComponent(destination, source.ClsModel.File, written, "classifier");
            CopyVerifiedComponent(destination, source.RecModel.File, written, "recognizer");
            CopyVerifiedComponent(destination, source.Dictionary, written, "dictionary");
        }
        catch
        {
            try
            {
                if (Directory.Exists(destination))
                {
                    Directory.Delete(destination, recursive: true);
                }
            }
            catch
            {
                // Preserve the snapshot verification error. The outer owned
                // stage cleanup will retry removal.
            }
            throw;
        }
    }

    private static void CopyVerifiedComponent(
        string destination,
        PaddleOcrFileRecord expected,
        HashSet<string> written,
        string role)
    {
        var bytes = RecipientDerivedCropFileContract.ReadRegularFile(
            expected.FullPath,
            $"Source Paddle OCR {role} snapshot");
        VerifyBoundBytes(
            bytes,
            expected.Sha256,
            expected.SizeBytes,
            $"Source Paddle OCR {role}");
        WriteSnapshotFile(destination, expected.RelativePath, bytes, written, role);
    }

    internal static void VerifyBoundBytes(
        byte[] bytes,
        string expectedSha256,
        long? expectedSizeBytes,
        string description)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        RecipientDerivedCropHash.RequireLowerSha256(
            expectedSha256,
            $"{description} expected SHA-256");
        if ((expectedSizeBytes is { } size && bytes.LongLength != size)
            || !string.Equals(
                RecipientDerivedCropHash.Sha256(bytes),
                expectedSha256,
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"{description} changed before private snapshot");
        }
    }

    private static void WriteSnapshotFile(
        string destination,
        string relativePath,
        byte[] bytes,
        HashSet<string> written,
        string description)
    {
        var path = RecipientDerivedCropFileContract.ResolveContainedFile(
            destination,
            relativePath,
            $"Private Paddle OCR {description} snapshot");
        if (!written.Add(path))
        {
            throw new InvalidOperationException(
                $"Duplicate private Paddle OCR snapshot path: {relativePath}");
        }
        var parent = Path.GetDirectoryName(path)
            ?? throw new InvalidOperationException(
                $"Private Paddle OCR snapshot path has no parent: {relativePath}");
        Directory.CreateDirectory(parent);
        RecipientDerivedCropFileContract.RequireNoReparseDirectoryChain(
            parent,
            $"Private Paddle OCR {description} snapshot parent");
        using (var stream = new FileStream(
            path,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None))
        {
            stream.Write(bytes);
            stream.Flush(flushToDisk: true);
        }
        var writtenIdentity = RecipientDerivedCropFileIdentity.FromFile(
            path,
            $"Private Paddle OCR {description} snapshot");
        if (writtenIdentity.SizeBytes != bytes.LongLength
            || !string.Equals(
                writtenIdentity.Sha256,
                RecipientDerivedCropHash.Sha256(bytes),
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Private Paddle OCR {description} snapshot differs after write");
        }
    }
}

internal sealed record RecipientDerivedCropBundleEvidence(
    string Directory,
    RecipientDerivedCropFileEvidence Contract,
    string SourceAuditContractSha256,
    long PackageSizeBytes,
    RecipientDerivedCropFileEvidence Detector,
    RecipientDerivedCropFileEvidence Classifier,
    RecipientDerivedCropFileEvidence Recognizer,
    RecipientDerivedCropFileEvidence Dictionary)
{
    public bool ContentEquals(RecipientDerivedCropBundleEvidence other)
    {
        ArgumentNullException.ThrowIfNull(other);
        static bool SameFile(
            RecipientDerivedCropFileEvidence left,
            RecipientDerivedCropFileEvidence right)
        {
            return string.Equals(left.Sha256, right.Sha256, StringComparison.Ordinal)
                && left.SizeBytes == right.SizeBytes;
        }

        return string.Equals(
                SourceAuditContractSha256,
                other.SourceAuditContractSha256,
                StringComparison.Ordinal)
            && PackageSizeBytes == other.PackageSizeBytes
            && SameFile(Contract, other.Contract)
            && SameFile(Detector, other.Detector)
            && SameFile(Classifier, other.Classifier)
            && SameFile(Recognizer, other.Recognizer)
            && SameFile(Dictionary, other.Dictionary);
    }

    public static RecipientDerivedCropBundleEvidence From(PaddleOcrDeliveryBundle bundle)
    {
        RecipientDerivedCropFileContract.RequireRegularNonReparseDirectory(
            bundle.BundleDirectory,
            "Paddle OCR delivery directory");
        var contract = RecipientDerivedCropFileIdentity.FromFile(
            bundle.ContractPath,
            "Paddle OCR delivery contract");
        if (!string.Equals(contract.Sha256, bundle.ContractSha256, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Paddle OCR delivery contract hash changed after verification");
        }
        return new RecipientDerivedCropBundleEvidence(
            Directory: Path.GetFullPath(bundle.BundleDirectory),
            Contract: FromIdentity(contract),
            SourceAuditContractSha256: bundle.SourceAuditContractSha256,
            PackageSizeBytes: bundle.PackageSizeBytes,
            Detector: FromFile(bundle.DetModel.File, "Paddle OCR detector"),
            Classifier: FromFile(bundle.ClsModel.File, "Paddle OCR classifier"),
            Recognizer: FromFile(bundle.RecModel.File, "Paddle OCR recognizer"),
            Dictionary: FromFile(bundle.Dictionary, "Paddle OCR dictionary"));
    }

    private static RecipientDerivedCropFileEvidence FromFile(
        PaddleOcrFileRecord file,
        string description)
    {
        RecipientDerivedCropFileContract.RequireRegularNonReparseFile(file.FullPath, description);
        var identity = RecipientDerivedCropFileIdentity.FromFile(file.FullPath, description);
        if (!string.Equals(identity.Sha256, file.Sha256, StringComparison.Ordinal)
            || identity.SizeBytes != file.SizeBytes)
        {
            throw new InvalidOperationException($"{description} identity changed after verification");
        }
        return FromIdentity(identity);
    }

    private static RecipientDerivedCropFileEvidence FromIdentity(
        RecipientDerivedCropFileIdentity identity)
    {
        return new RecipientDerivedCropFileEvidence(
            identity.Path,
            identity.Sha256,
            identity.SizeBytes);
    }
}

internal sealed record RecipientDerivedCropInputPlanEvidence(
    string Directory,
    string PlanSummaryPath,
    string PlanSummarySha256,
    long PlanSummarySizeBytes,
    string Path,
    string Sha256,
    long SizeBytes,
    int Records,
    string InputsPath,
    string InputsSha256,
    long InputsSizeBytes,
    string FilterContractPath,
    string FilterContractSha256,
    long FilterContractSizeBytes)
{
    public static RecipientDerivedCropInputPlanEvidence From(
        RecipientDerivedCropPlanSelection selection)
    {
        return new RecipientDerivedCropInputPlanEvidence(
            selection.Directory,
            selection.Summary.Path,
            selection.Summary.Sha256,
            selection.Summary.SizeBytes,
            selection.Plans.Path,
            selection.Plans.Sha256,
            selection.Plans.SizeBytes,
            selection.Records.Count,
            selection.Inputs.Path,
            selection.Inputs.Sha256,
            selection.Inputs.SizeBytes,
            selection.FilterContract.Path,
            selection.FilterContract.Sha256,
            selection.FilterContract.SizeBytes);
    }
}

internal sealed record RecipientDerivedCropArtifactEvidence(
    string Path,
    string Sha256,
    long SizeBytes,
    int Records)
{
    public static RecipientDerivedCropArtifactEvidence From(
        string file,
        string path,
        int records)
    {
        var identity = RecipientDerivedCropFileIdentity.FromFile(file, "Derived-crop records artifact");
        return new RecipientDerivedCropArtifactEvidence(
            path,
            identity.Sha256,
            identity.SizeBytes,
            records);
    }
}

internal sealed record RecipientDerivedCropLatencyDistribution(
    int Count,
    double Mean,
    double P50,
    double P95,
    double P99,
    double Max)
{
    public static RecipientDerivedCropLatencyDistribution From(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(value => value).ToArray();
        if (sorted.Length == 0
            || sorted.Any(value => !double.IsFinite(value) || value < 0.0))
        {
            throw new InvalidOperationException("Recipient derived-crop latency evidence is empty or invalid");
        }
        return new RecipientDerivedCropLatencyDistribution(
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

internal sealed record RecipientDerivedCropLatencySummary(
    RecipientDerivedCropLatencyDistribution ImageLoad,
    RecipientDerivedCropLatencyDistribution Rectification,
    RecipientDerivedCropLatencyDistribution Crop4LayoutOcr,
    RecipientDerivedCropLatencyDistribution Crop5LayoutOcr,
    RecipientDerivedCropLatencyDistribution Total)
{
    public static RecipientDerivedCropLatencySummary From(
        IReadOnlyCollection<RecipientDerivedCropTiming> timings)
    {
        return new RecipientDerivedCropLatencySummary(
            RecipientDerivedCropLatencyDistribution.From(timings.Select(value => value.ImageLoad)),
            RecipientDerivedCropLatencyDistribution.From(timings.Select(value => value.Rectification)),
            RecipientDerivedCropLatencyDistribution.From(timings.Select(value => value.Crop4LayoutOcr)),
            RecipientDerivedCropLatencyDistribution.From(timings.Select(value => value.Crop5LayoutOcr)),
            RecipientDerivedCropLatencyDistribution.From(timings.Select(value => value.Total)));
    }
}

internal sealed record RecipientDerivedCropArtifacts(
    RecipientDerivedCropArtifactEvidence Records);

internal sealed record RecipientDerivedCropSummary(
    int SchemaVersion,
    string Kind,
    bool DiagnosticOnly,
    bool FormalDeliveryGate,
    bool CandidateWriteEnabled,
    bool ProductionOutputChanged,
    bool AccuracyClaimed,
    bool TruthUsedForCandidateSelection,
    bool OcrRerun,
    int ExpectedRecords,
    int Records,
    int Errors,
    string ExecutionProvider,
    string Rectification,
    IReadOnlyList<string> CropNames,
    string QuadCoordinateSpace,
    string ConfidenceSemantics,
    float PaddleDropScore,
    RecipientDerivedCropInputPlanEvidence InputPlan,
    RecipientDerivedCropBundleEvidence PaddleBundle,
    RecipientDerivedCropLatencySummary LatencyMs,
    RecipientDerivedCropArtifacts Artifacts);
