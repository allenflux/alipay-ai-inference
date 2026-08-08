using System.Text;
using System.Text.Json;
using OpenCvSharp;

internal static class Program
{
    private static int Main()
    {
        try
        {
            VerifyAcceptedProjectionMatchesLegacySemantics();
            VerifyEmptyAndInvalidLayoutContracts();
            VerifyFrozenInputContract();
            VerifyOptionsAreDiagnosticOnly();
            VerifyJsonHasNoProductionFields();
            VerifyFreshOutputContract();
            VerifyOutputIsDisjointFromBundle();
            Console.WriteLine(
                "PASS: raw Paddle layout pairing, legacy accepted-line projection, frozen-339 input, "
                + "CPU-only diagnostic schema, and fresh-output contracts are closed.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void VerifyAcceptedProjectionMatchesLegacySemantics()
    {
        const float dropScore = 0.5f;
        var boxes = new[]
        {
            Box(10, 20, 110, 40),
            Box(15, 50, 115, 70),
            Box(20, 80, 120, 100),
        };
        PaddleOcrLine?[] decoded =
        [
            new PaddleOcrLine("  first   value ", 0.91f),
            new PaddleOcrLine("   ", 0.81f),
            new PaddleOcrLine("rejected", 0.49f),
        ];
        var expected = LegacyProjection(decoded, dropScore);
        var actual = PaddleOcrEngine.AssembleLayoutDiagnostic(boxes, decoded, dropScore);

        AssertEqual(expected.Text, actual.Text, "accepted aggregate text");
        AssertNullableFloatBitsEqual(expected.Confidence, actual.Confidence, "accepted aggregate confidence");
        AssertEqual(expected.Lines.Count, actual.AcceptedLines.Count, "accepted line count");
        for (var index = 0; index < expected.Lines.Count; index++)
        {
            AssertEqual(expected.Lines[index].Text, actual.AcceptedLines[index].Text, $"accepted line {index} text");
            AssertFloatBitsEqual(expected.Lines[index].Confidence, actual.AcceptedLines[index].Confidence, $"accepted line {index} confidence");
        }
        AssertEqual(3, actual.Lines.Count, "raw layout line count");
        Assert(actual.Lines[0].PassesDropScore, "first line must pass drop score");
        Assert(actual.Lines[1].PassesDropScore, "clean-empty line must still pass and contribute confidence");
        Assert(!actual.Lines[2].PassesDropScore, "third line must remain visible but rejected");
        AssertEqual("rejected", actual.Lines[2].Text, "rejected line text remains raw evidence");
        AssertEqual(10.0f, actual.Lines[0].Quad[0].X, "quad TL x");
        AssertEqual(20.0f, actual.Lines[0].Quad[0].Y, "quad TL y");
        AssertEqual(110.0f, actual.Lines[0].Quad[1].X, "quad TR x");
        AssertEqual(40.0f, actual.Lines[0].Quad[2].Y, "quad BR y");
        AssertEqual(10.0f, actual.Lines[0].Quad[3].X, "quad BL x");

        // The diagnostic record must own an immutable copy rather than expose
        // the mutable OpenCV array returned by DB post-processing.
        boxes[0][0] = new Point2f(999, 999);
        AssertEqual(10.0f, actual.Lines[0].Quad[0].X, "quad copy remains immutable");

        var recordLines = LayoutShadowProgram.BuildLayoutLines(actual, 121, 101);
        AssertEqual(3, recordLines.Count, "record line count");
        for (var index = 0; index < recordLines.Count; index++)
        {
            AssertEqual(index, recordLines[index].Index, $"record line {index} index");
            AssertEqual(actual.Lines[index].Text, recordLines[index].Text, $"record line {index} text binding");
            AssertFloatBitsEqual(
                actual.Lines[index].Confidence,
                recordLines[index].Confidence,
                $"record line {index} confidence binding");
        }
        AssertFails(
            () => LayoutShadowProgram.BuildLayoutLines(actual, 120, 101),
            "outside rectified image bounds");

        var wrongAccepted = actual with
        {
            AcceptedLines =
            [
                new PaddleOcrLine("wrong", actual.AcceptedLines[0].Confidence),
                actual.AcceptedLines[1],
            ],
        };
        AssertFails(
            () => LayoutShadowProgram.BuildLayoutLines(wrongAccepted, 121, 101),
            "not index-bound to accepted CTC output");
    }

    private static void VerifyEmptyAndInvalidLayoutContracts()
    {
        var empty = PaddleOcrEngine.AssembleLayoutDiagnostic(
            Array.Empty<Point2f[]>(),
            Array.Empty<PaddleOcrLine?>(),
            0.5f);
        AssertEqual(string.Empty, empty.Text, "empty aggregate text");
        Assert(empty.Confidence is null, "empty aggregate confidence must be null");
        AssertEqual(0, empty.AcceptedLines.Count, "empty accepted lines");
        AssertEqual(0, empty.Lines.Count, "empty raw lines");

        AssertFails(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                [Box(0, 0, 10, 10)],
                Array.Empty<PaddleOcrLine?>(),
                0.5f),
            "box/line count differs");
        AssertFails(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                [new[] { new Point2f(0, 0) }],
                [new PaddleOcrLine("x", 0.9f)],
                0.5f),
            "not a finite quadrilateral");
        AssertFails(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                [new[]
                {
                    new Point2f(float.NaN, 0),
                    new Point2f(1, 0),
                    new Point2f(1, 1),
                    new Point2f(0, 1),
                }],
                [new PaddleOcrLine("x", 0.9f)],
                0.5f),
            "not a finite quadrilateral");
        AssertFails(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                [Box(0, 0, 10, 10)],
                new PaddleOcrLine?[] { null },
                0.5f),
            "was not decoded");
        AssertFails(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                [Box(0, 0, 10, 10)],
                [new PaddleOcrLine("x", float.PositiveInfinity)],
                0.5f),
            "non-finite confidence");
        AssertFails(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                Array.Empty<Point2f[]>(),
                Array.Empty<PaddleOcrLine?>(),
                float.NaN),
            "drop score must be finite");
    }

    private static void VerifyFrozenInputContract()
    {
        var root = Path.Combine(Path.GetTempPath(), $"layout-shadow-contract-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var sources = Enumerable.Range(0, LayoutShadowProgram.ExpectedRecordCount)
                .Select(index => Path.Combine(root, $"receipt-{index:D3}.jpg"))
                .ToArray();
            foreach (var source in sources)
            {
                File.WriteAllBytes(source, Array.Empty<byte>());
            }
            var list = Path.Combine(root, "inputs.txt");
            WriteList(list, sources);
            var hash = LayoutShadowHash.Sha256(File.ReadAllBytes(list));
            var selection = LayoutShadowInputContract.Load(list, hash);
            AssertEqual(LayoutShadowProgram.ExpectedRecordCount, selection.Sources.Count, "frozen input count");
            AssertEqual(hash, selection.Sha256, "frozen input hash");
            AssertEqual(Path.GetFullPath(sources[0]), selection.Sources[0], "frozen order first");
            AssertEqual(Path.GetFullPath(sources[^1]), selection.Sources[^1], "frozen order last");

            var sourceEvidence = sources
                .Select(source =>
                {
                    var identity = LayoutShadowHash.Sha256FileEvidence(source);
                    return new LayoutShadowSourceEvidence(
                        source,
                        identity.Sha256,
                        identity.SizeBytes);
                })
                .ToArray();
            LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence);

            File.WriteAllBytes(sources[0], [1]);
            AssertFails(
                () => LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence),
                "source changed while");
            File.WriteAllBytes(sources[0], Array.Empty<byte>());
            LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence);

            File.AppendAllText(list, "# changed after load\n", Encoding.UTF8);
            AssertFails(
                () => LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence),
                "input list changed while");
            WriteList(list, sources);
            LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence);

            AssertFails(
                () => LayoutShadowInputContract.Load(list, new string('0', 64)),
                "SHA-256 differs");
            AssertFails(
                () => LayoutShadowInputContract.Load(list, "A" + hash[1..]),
                "lowercase hexadecimal");

            var shortList = Path.Combine(root, "short.txt");
            WriteList(shortList, sources[..^1]);
            AssertFails(
                () => LayoutShadowInputContract.Load(
                    shortList,
                    LayoutShadowHash.Sha256(File.ReadAllBytes(shortList))),
                "exactly 339");

            var duplicateList = Path.Combine(root, "duplicate.txt");
            var duplicates = sources.ToArray();
            duplicates[^1] = duplicates[0];
            WriteList(duplicateList, duplicates);
            AssertFails(
                () => LayoutShadowInputContract.Load(
                    duplicateList,
                    LayoutShadowHash.Sha256(File.ReadAllBytes(duplicateList))),
                "Duplicate layout shadow source");

            var blankList = Path.Combine(root, "blank.txt");
            var withBlank = sources.ToList();
            withBlank.Insert(1, string.Empty);
            WriteList(blankList, withBlank);
            AssertFails(
                () => LayoutShadowInputContract.Load(
                    blankList,
                    LayoutShadowHash.Sha256(File.ReadAllBytes(blankList))),
                "blank line");

            var commentList = Path.Combine(root, "comment.txt");
            var withComment = sources.ToList();
            withComment.Insert(1, "# forbidden");
            WriteList(commentList, withComment);
            AssertFails(
                () => LayoutShadowInputContract.Load(
                    commentList,
                    LayoutShadowHash.Sha256(File.ReadAllBytes(commentList))),
                "must not contain comments");

            var relativeList = Path.Combine(root, "relative.txt");
            var withRelative = sources.ToArray();
            withRelative[0] = "receipt-relative.jpg";
            WriteList(relativeList, withRelative);
            AssertFails(
                () => LayoutShadowInputContract.Load(
                    relativeList,
                    LayoutShadowHash.Sha256(File.ReadAllBytes(relativeList))),
                "must be an absolute path");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void VerifyOptionsAreDiagnosticOnly()
    {
        var parsed = LayoutShadowOptions.Parse(
        [
            "--bundle", "bundle",
            "--input-list", "inputs.txt",
            "--input-list-sha256", new string('a', 64),
            "--output", "output",
        ]);
        AssertEqual("bundle", parsed.Bundle, "bundle option");
        AssertFails(
            () => LayoutShadowOptions.Parse(
            [
                "--bundle", "bundle",
                "--input-list", "inputs.txt",
                "--input-list-sha256", new string('a', 64),
                "--output", "output",
                "--device", "cuda:0",
            ]),
            "Unknown layout shadow argument");
        AssertFails(
            () => LayoutShadowOptions.Parse(
            [
                "--bundle", "one",
                "--bundle", "two",
                "--input-list", "inputs.txt",
                "--input-list-sha256", new string('a', 64),
                "--output", "output",
            ]),
            "Duplicate layout shadow argument");
    }

    private static void VerifyJsonHasNoProductionFields()
    {
        var geometry = new RectificationGeometry(
            new ImageSize(10, 20),
            new ImageSize(10, 20),
            ReceiptRectifier.MaxSide1600Mode,
            0,
            false,
            [[0, 0], [9, 0], [9, 19], [0, 19]],
            Identity(),
            Identity());
        var record = new LayoutShadowRecord(
            1,
            LayoutShadowProgram.RecordKind,
            true,
            false,
            false,
            0,
            Path.GetFullPath("fixture.jpg"),
            new string('a', 64),
            123,
            "cpu",
            geometry,
            LayoutShadowProgram.QuadCoordinateSpace,
            LayoutShadowProgram.QuadNormalization,
            LayoutShadowProgram.ConfidenceSemantics,
            "转账成功",
            0.9f,
            1,
            1,
            [new LayoutShadowLine(
                0,
                "转账成功",
                0.9f,
                true,
                [[0, 0], [9, 0], [9, 4], [0, 4]],
                [[0, 0], [1, 0], [1, 0.25f], [0, 0.25f]])],
            new LayoutShadowTiming(1, 2, 3, 6));
        var json = LayoutShadowJson.Serialize(record);
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        Assert(root.GetProperty("diagnostic_only").GetBoolean(), "record diagnostic_only");
        Assert(!root.GetProperty("formal_delivery_gate").GetBoolean(), "record formal_delivery_gate");
        Assert(!root.GetProperty("candidate_write_enabled").GetBoolean(), "record candidate_write_enabled");
        AssertEqual("cpu", root.GetProperty("execution_provider").GetString(), "record CPU provider");
        foreach (var forbidden in new[] { "fields", "candidate", "delivery_value", "detector", "device" })
        {
            Assert(!root.TryGetProperty(forbidden, out _), $"record must omit {forbidden}");
        }
        var line = root.GetProperty("lines")[0];
        AssertEqual(4, line.GetProperty("quad_rectified").GetArrayLength(), "record raw quad");
        AssertEqual(4, line.GetProperty("quad_rectified_normalized").GetArrayLength(), "record normalized quad");

        var hash = new string('b', 64);
        var component = new LayoutShadowFileEvidence("component.bin", hash, 1);
        var latency = new LayoutShadowLatencyDistribution(1, 1, 1, 1, 1, 1);
        var summary = new LayoutShadowSummary(
            1,
            LayoutShadowProgram.SummaryKind,
            true,
            false,
            false,
            LayoutShadowProgram.ExpectedRecordCount,
            LayoutShadowProgram.ExpectedRecordCount,
            0,
            "cpu",
            ReceiptRectifier.MaxSide1600Mode,
            LayoutShadowProgram.QuadCoordinateSpace,
            LayoutShadowProgram.QuadNormalization,
            LayoutShadowProgram.ConfidenceSemantics,
            0.5f,
            new LayoutShadowInputEvidence(
                Path.GetFullPath("inputs.txt"),
                hash,
                10,
                LayoutShadowProgram.ExpectedRecordCount),
            new LayoutShadowBundleEvidence(
                Path.GetFullPath("bundle"),
                Path.GetFullPath("bundle/paddle_ocr_delivery.contract.json"),
                hash,
                hash,
                4,
                component,
                component,
                component,
                component),
            new LayoutShadowStageLatencySummary(latency, latency, latency, latency),
            new LayoutShadowArtifacts(
                new LayoutShadowArtifactEvidence("records.jsonl", hash, 100)));
        using var summaryDocument = JsonDocument.Parse(
            LayoutShadowJson.Serialize(summary));
        var summaryRoot = summaryDocument.RootElement;
        Assert(summaryRoot.GetProperty("diagnostic_only").GetBoolean(), "summary diagnostic_only");
        Assert(!summaryRoot.GetProperty("formal_delivery_gate").GetBoolean(), "summary formal_delivery_gate");
        Assert(!summaryRoot.GetProperty("candidate_write_enabled").GetBoolean(), "summary candidate_write_enabled");
        foreach (var forbidden in new[] { "fields", "candidate", "delivery_value", "detector", "device" })
        {
            Assert(!summaryRoot.TryGetProperty(forbidden, out _), $"summary must omit {forbidden}");
        }
    }

    private static void VerifyFreshOutputContract()
    {
        var root = Path.Combine(Path.GetTempPath(), $"layout-shadow-output-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var output = Path.Combine(root, "fresh");
            var resolved = LayoutShadowOutputContract.ResolveFreshOutput(output);
            AssertEqual(Path.GetFullPath(output), resolved.FullPath, "fresh output path");
            Directory.CreateDirectory(output);
            AssertFails(
                () => LayoutShadowOutputContract.ResolveFreshOutput(output),
                "Refusing to overwrite");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void VerifyOutputIsDisjointFromBundle()
    {
        var root = Path.Combine(Path.GetTempPath(), $"layout-shadow-disjoint-{Guid.NewGuid():N}");
        var bundle = Path.Combine(root, "bundle");
        Directory.CreateDirectory(bundle);
        try
        {
            var inside = LayoutShadowOutputContract.ResolveFreshOutput(
                Path.Combine(bundle, "forbidden-output"));
            AssertFails(
                () => LayoutShadowOutputContract.RequireDisjointFromBundle(
                    inside,
                    bundle),
                "must be disjoint");

            var sibling = LayoutShadowOutputContract.ResolveFreshOutput(
                Path.Combine(root, "allowed-output"));
            LayoutShadowOutputContract.RequireDisjointFromBundle(sibling, bundle);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static PaddleOcrReadResult LegacyProjection(
        IReadOnlyList<PaddleOcrLine?> decoded,
        float dropScore)
    {
        var accepted = decoded
            .Where(line => line is not null && line.Confidence >= dropScore)
            .Select(line => line!)
            .ToList();
        if (accepted.Count == 0)
        {
            return new PaddleOcrReadResult(string.Empty, null, Array.Empty<PaddleOcrLine>());
        }
        return new PaddleOcrReadResult(
            string.Join(" ", accepted
                .Select(line => ReceiptFieldNormalizer.CleanText(line.Text))
                .Where(text => text.Length > 0)),
            accepted.Average(line => line.Confidence),
            accepted);
    }

    private static Point2f[] Box(float x1, float y1, float x2, float y2)
    {
        return
        [
            new Point2f(x1, y1),
            new Point2f(x2, y1),
            new Point2f(x2, y2),
            new Point2f(x1, y2),
        ];
    }

    private static double[][] Identity()
    {
        return
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ];
    }

    private static void WriteList(string path, IEnumerable<string> values)
    {
        File.WriteAllLines(
            path,
            values,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    private static void AssertFails(Action action, string messageFragment)
    {
        try
        {
            action();
        }
        catch (Exception error) when (error.Message.Contains(messageFragment, StringComparison.Ordinal))
        {
            return;
        }
        throw new InvalidOperationException($"Expected failure containing '{messageFragment}'");
    }

    private static void AssertNullableFloatBitsEqual(float? expected, float? actual, string label)
    {
        if (expected is null || actual is null)
        {
            if (expected != actual)
            {
                throw new InvalidOperationException($"{label}: expected={expected}, actual={actual}");
            }
            return;
        }
        AssertFloatBitsEqual(expected.Value, actual.Value, label);
    }

    private static void AssertFloatBitsEqual(float expected, float actual, string label)
    {
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
        {
            throw new InvalidOperationException($"{label}: expected={expected:R}, actual={actual:R}");
        }
    }

    private static void Assert<T>(bool condition, T message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message?.ToString());
        }
    }

    private static void AssertEqual<T>(T expected, T actual, string label)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            throw new InvalidOperationException($"{label}: expected={expected}, actual={actual}");
        }
    }
}
