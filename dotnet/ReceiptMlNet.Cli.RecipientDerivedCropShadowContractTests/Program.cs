using System.Text;
using System.Text.Json;

internal static class Program
{
    private static int Main()
    {
        try
        {
            VerifyOptionsRequireExternallyBoundPlanSummary();
            VerifyFrozenPlanAndInputContract();
            VerifyPlanAndSourceMutationAreRejected();
            VerifyGlobalGateFailureIsRejected();
            VerifyCanonicalCropGeometry();
            VerifyPythonCanonicalPlanId();
            VerifyModelByteSnapshotIsHashBoundAndCloned();
            VerifyContractAndDictionarySwapAreRejected();
            VerifyRawLayoutLineCoordinatesAndDropScore();
            VerifyDiagnosticJsonHasNoFieldCandidate();
            VerifyFreshDisjointOutputContract();
            Console.WriteLine(
                "PASS: frozen-63 plan/input/source binding, canonical crop4/crop5, raw layout quads, "
                + "CPU-only diagnostic schema, TOCTOU, and fresh atomic-output contracts are closed.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void VerifyOptionsRequireExternallyBoundPlanSummary()
    {
        var hash = new string('a', 64);
        var options = RecipientDerivedCropShadowOptions.Parse(
        [
            "--bundle", "bundle",
            "--plan", "plan",
            "--plan-summary-sha256", hash,
            "--output", "output",
        ]);
        AssertEqual("bundle", options.Bundle, "bundle option");
        AssertEqual("plan", options.PlanDirectory, "plan option");
        AssertEqual(hash, options.PlanSummarySha256, "plan summary hash option");
        AssertEqual("output", options.Output, "output option");
        AssertFails(
            () => RecipientDerivedCropShadowOptions.Parse(
            [
                "--bundle", "bundle",
                "--plan", "plan",
                "--output", "output",
            ]),
            "Usage:");
        AssertFails(
            () => RecipientDerivedCropShadowOptions.Parse(
            [
                "--bundle", "bundle",
                "--plan", "plan",
                "--plan-summary-sha256", "NOT-A-HASH",
                "--output", "output",
            ]),
            "64 lowercase");
        AssertFails(
            () => RecipientDerivedCropShadowOptions.Parse(
            [
                "--bundle", "bundle",
                "--plan", "plan",
                "--plan-summary-sha256", hash,
                "--device", "cuda:0",
                "--output", "output",
            ]),
            "Unknown");
    }

    private static void VerifyFrozenPlanAndInputContract()
    {
        using var fixture = PlanFixture.Create();
        var selection = RecipientDerivedCropPlanContract.Load(
            fixture.PlanDirectory,
            fixture.SummarySha256);
        AssertEqual(63, selection.Records.Count, "plan record count");
        AssertEqual(63, selection.Records.Select(record => record.Source).Distinct().Count(), "source count");
        AssertEqual(63, selection.Records.Select(record => record.PlanId).Distinct().Count(), "plan id count");
        AssertEqual(
            RecipientDerivedCropShadowProgram.Crop4,
            selection.Records[0].Crops[0].Name,
            "crop4 name");
        AssertEqual(
            RecipientDerivedCropShadowProgram.Crop5,
            selection.Records[0].Crops[1].Name,
            "crop5 name");
        RecipientDerivedCropPlanContract.VerifyUnchanged(selection);
    }

    private static void VerifyPlanAndSourceMutationAreRejected()
    {
        using (var fixture = PlanFixture.Create())
        {
            var selection = RecipientDerivedCropPlanContract.Load(
                fixture.PlanDirectory,
                fixture.SummarySha256);
            File.AppendAllText(
                Path.Combine(fixture.PlanDirectory, "inputs.txt"),
                "mutated",
                new UTF8Encoding(false));
            AssertFails(
                () => RecipientDerivedCropPlanContract.VerifyUnchanged(selection),
                "inputs artifact changed");
        }
        using (var fixture = PlanFixture.Create())
        {
            var selection = RecipientDerivedCropPlanContract.Load(
                fixture.PlanDirectory,
                fixture.SummarySha256);
            File.AppendAllText(
                selection.Records[0].Source,
                "mutated",
                new UTF8Encoding(false));
            AssertFails(
                () => RecipientDerivedCropPlanContract.VerifyUnchanged(selection),
                "source identity differs");
        }
        using (var fixture = PlanFixture.Create())
        {
            AssertFails(
                () => RecipientDerivedCropPlanContract.Load(
                    fixture.PlanDirectory,
                    new string('0', 64)),
                "summary SHA-256 differs");
        }
    }

    private static void VerifyGlobalGateFailureIsRejected()
    {
        using var fixture = PlanFixture.Create(firstRecordGateFailure: true);
        AssertFails(
            () => RecipientDerivedCropPlanContract.Load(
                fixture.PlanDirectory,
                fixture.SummarySha256),
            "does not preserve every global gate");
    }

    private static void VerifyCanonicalCropGeometry()
    {
        var geometry = new RecipientDerivedCropGeometry(
            Amount: [200.0, 300.0, 800.0, 500.0],
            Recipient: [100.0, 600.0, 900.0, 700.0],
            Payment: [200.0, 800.0, 800.0, 900.0]);
        var crops = RecipientDerivedCropPlanContract.DeriveCrops(
            geometry,
            new RecipientDerivedCropSize(1000, 1600));
        AssertSequenceEqual([280, 565, 964, 735], crops[0].Box.ToArray(), "canonical crop4");
        AssertSequenceEqual([360, 592, 964, 708], crops[1].Box.ToArray(), "canonical crop5");
        Assert(crops[0].Box.Left < crops[1].Box.Left, "crop contexts must be horizontally distinct");
    }

    private static void VerifyPythonCanonicalPlanId()
    {
        using var document = JsonDocument.Parse(
            "{\"z\":200.0,\"a\":\"商户\",\"plan_id\":\"ignored\"}");
        AssertEqual(
            "e6f70387ee350638c436b281af8524bb4cd0f64ba4353dcb6b2c905085f5aab0",
            RecipientDerivedCropPlanContract.CanonicalPlanId(document.RootElement),
            "Python sort_keys/ensure_ascii=False canonical plan id");
    }

    private static void VerifyModelByteSnapshotIsHashBoundAndCloned()
    {
        var bytes = Encoding.UTF8.GetBytes("verified-model-bytes");
        var expected = new PaddleOcrFileRecord(
            RelativePath: "model.onnx",
            FullPath: Path.GetFullPath("model.onnx"),
            Sha256: RecipientDerivedCropHash.Sha256(bytes),
            SizeBytes: bytes.LongLength);
        var snapshot = PaddleOcrCpuModelSnapshot.VerifyAndClone(bytes, expected, "fixture");
        bytes[0] ^= 0x01;
        Assert(
            !snapshot.SequenceEqual(bytes),
            "model snapshot must own a clone independent of caller mutation");
        AssertFails(
            () => PaddleOcrCpuModelSnapshot.VerifyAndClone(bytes, expected, "fixture"),
            "differs from the verified delivery contract");
    }

    private static void VerifyContractAndDictionarySwapAreRejected()
    {
        var contract = Encoding.UTF8.GetBytes("{\"kind\":\"verified-contract\"}");
        var contractSha256 = RecipientDerivedCropHash.Sha256(contract);
        RecipientDerivedCropBundleSnapshot.VerifyBoundBytes(
            contract,
            contractSha256,
            expectedSizeBytes: null,
            "contract fixture");
        var swappedContract = contract.ToArray();
        swappedContract[^2] ^= 0x01;
        AssertFails(
            () => RecipientDerivedCropBundleSnapshot.VerifyBoundBytes(
                swappedContract,
                contractSha256,
                expectedSizeBytes: null,
                "contract fixture"),
            "changed before private snapshot");

        var dictionary = Encoding.UTF8.GetBytes("商\n户\n");
        var dictionarySha256 = RecipientDerivedCropHash.Sha256(dictionary);
        RecipientDerivedCropBundleSnapshot.VerifyBoundBytes(
            dictionary,
            dictionarySha256,
            dictionary.LongLength,
            "dictionary fixture");
        var swappedDictionary = Encoding.UTF8.GetBytes("商\n甲\n");
        AssertFails(
            () => RecipientDerivedCropBundleSnapshot.VerifyBoundBytes(
                swappedDictionary,
                dictionarySha256,
                dictionary.LongLength,
                "dictionary fixture"),
            "changed before private snapshot");
        AssertFails(
            () => RecipientDerivedCropBundleSnapshot.VerifyBoundBytes(
                dictionary,
                dictionarySha256,
                dictionary.LongLength + 1,
                "dictionary fixture"),
            "changed before private snapshot");
    }

    private static void VerifyRawLayoutLineCoordinatesAndDropScore()
    {
        var crop = new RecipientDerivedCropPlanCrop(
            RecipientDerivedCropShadowProgram.Crop4,
            new RecipientDerivedCropBox(100, 200, 300, 260));
        var line = new PaddleOcrLine("商户甲", 0.91f);
        var layout = new PaddleOcrLayoutLine(
            line.Text,
            line.Confidence,
            [
                new PaddleOcrLayoutPoint(10, 5),
                new PaddleOcrLayoutPoint(90, 5),
                new PaddleOcrLayoutPoint(90, 25),
                new PaddleOcrLayoutPoint(10, 25),
            ],
            PassesDropScore: true);
        var read = new PaddleOcrLayoutReadResult(
            Text: line.Text,
            Confidence: line.Confidence,
            AcceptedLines: [line],
            Lines: [layout]);
        var output = RecipientDerivedCropShadowProgram.BuildLines(read, crop, dropScore: 0.5f);
        AssertEqual(1, output.Count, "raw line count");
        AssertEqual("商户甲", output[0].Text, "raw line text");
        AssertFloatBitsEqual(10.0f, output[0].QuadCrop[0][0], "crop quad x");
        AssertFloatBitsEqual(5.0f, output[0].QuadCrop[0][1], "crop quad y");
        AssertFloatBitsEqual(110.0f, output[0].QuadRectified[0][0], "rectified quad x");
        AssertFloatBitsEqual(205.0f, output[0].QuadRectified[0][1], "rectified quad y");

        var wrongDropFlag = read with
        {
            Lines =
            [
                layout with
                {
                    Confidence = 0.49f,
                    PassesDropScore = true,
                },
            ],
        };
        AssertFails(
            () => RecipientDerivedCropShadowProgram.BuildLines(wrongDropFlag, crop, 0.5f),
            "drop-score flag differs");
        var outside = read with
        {
            Lines =
            [
                layout with
                {
                    Quad =
                    [
                        new PaddleOcrLayoutPoint(202, 5),
                        new PaddleOcrLayoutPoint(203, 5),
                        new PaddleOcrLayoutPoint(203, 25),
                        new PaddleOcrLayoutPoint(202, 25),
                    ],
                },
            ],
        };
        AssertFails(
            () => RecipientDerivedCropShadowProgram.BuildLines(outside, crop, 0.5f),
            "escapes crop bounds");
        var bowTie = read with
        {
            Lines =
            [
                layout with
                {
                    Quad =
                    [
                        new PaddleOcrLayoutPoint(10, 5),
                        new PaddleOcrLayoutPoint(90, 25),
                        new PaddleOcrLayoutPoint(90, 5),
                        new PaddleOcrLayoutPoint(10, 25),
                    ],
                },
            ],
        };
        AssertFails(
            () => RecipientDerivedCropShadowProgram.BuildLines(bowTie, crop, 0.5f),
            "degenerate, non-convex");
    }

    private static void VerifyDiagnosticJsonHasNoFieldCandidate()
    {
        var record = new RecipientDerivedCropRecord(
            SchemaVersion: 1,
            Kind: RecipientDerivedCropShadowProgram.RecordKind,
            DiagnosticOnly: true,
            FormalDeliveryGate: false,
            CandidateWriteEnabled: false,
            ProductionOutputChanged: false,
            Index: 0,
            Source: Path.GetFullPath("fixture.jpg"),
            SourceImageSha256: new string('a', 64),
            SourceImageSizeBytes: 10,
            ExecutionProvider: "cpu",
            Rectification: RecipientDerivedCropShadowProgram.PlanRectification,
            RectifiedSize: new RecipientDerivedCropSize(1000, 1600),
            PlanId: new string('b', 64),
            QuadCoordinateSpace: RecipientDerivedCropShadowProgram.QuadCoordinateSpace,
            ConfidenceSemantics: RecipientDerivedCropShadowProgram.ConfidenceSemantics,
            Crops:
            [
                new RecipientDerivedCropLayout(
                    RecipientDerivedCropShadowProgram.Crop4,
                    [280, 565, 964, 735],
                    684,
                    170,
                    Array.Empty<RecipientDerivedCropLine>()),
                new RecipientDerivedCropLayout(
                    RecipientDerivedCropShadowProgram.Crop5,
                    [360, 592, 964, 708],
                    604,
                    116,
                    Array.Empty<RecipientDerivedCropLine>()),
            ],
            TimingMs: new RecipientDerivedCropTiming(1, 2, 3, 4, 10));
        using var json = JsonDocument.Parse(RecipientDerivedCropJson.Serialize(record));
        var root = json.RootElement;
        Assert(root.GetProperty("diagnostic_only").GetBoolean(), "record is diagnostic-only");
        Assert(!root.GetProperty("formal_delivery_gate").GetBoolean(), "record is not a delivery gate");
        Assert(!root.GetProperty("candidate_write_enabled").GetBoolean(), "candidate write remains disabled");
        Assert(!root.GetProperty("production_output_changed").GetBoolean(), "production output remains unchanged");
        AssertEqual("cpu", root.GetProperty("execution_provider").GetString(), "CPU provider evidence");
        Assert(!root.TryGetProperty("candidate", out _), "record must not contain candidate");
        Assert(!root.TryGetProperty("shadow_candidate", out _), "record must not contain shadow candidate");
        Assert(!root.TryGetProperty("fields", out _), "record must not contain fields");
        AssertEqual(2, root.GetProperty("crops").GetArrayLength(), "record crop count");
    }

    private static void VerifyFreshDisjointOutputContract()
    {
        var root = Path.Combine(Path.GetTempPath(), $"recipient-derived-output-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var outputPath = Path.Combine(root, "fresh-output");
            var output = RecipientDerivedCropOutputContract.ResolveFreshOutput(outputPath);
            AssertEqual(Path.GetFullPath(outputPath), output.FullPath, "fresh output path");
            Directory.CreateDirectory(outputPath);
            AssertFails(
                () => RecipientDerivedCropOutputContract.ResolveFreshOutput(outputPath),
                "Refusing to overwrite");
            var nested = new RecipientDerivedCropOutput(
                Path.Combine(root, "protected", "nested"),
                Path.Combine(root, "protected"),
                "nested");
            AssertFails(
                () => RecipientDerivedCropOutputContract.RequireDisjoint(
                    nested,
                    Path.Combine(root, "protected"),
                    "fixture"),
                "must be disjoint");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AssertFails(Action action, string expectedFragment)
    {
        try
        {
            action();
        }
        catch (Exception error)
        {
            if (!error.Message.Contains(expectedFragment, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    $"Expected failure containing '{expectedFragment}', got: {error.Message}",
                    error);
            }
            return;
        }
        throw new InvalidOperationException($"Expected failure containing '{expectedFragment}'");
    }

    private static void Assert(bool condition, string description)
    {
        if (!condition)
        {
            throw new InvalidOperationException($"Assertion failed: {description}");
        }
    }

    private static void AssertEqual<T>(T expected, T actual, string description)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            throw new InvalidOperationException(
                $"Assertion failed for {description}: expected={expected}, actual={actual}");
        }
    }

    private static void AssertSequenceEqual(
        IReadOnlyList<int> expected,
        IReadOnlyList<int> actual,
        string description)
    {
        if (!expected.SequenceEqual(actual))
        {
            throw new InvalidOperationException(
                $"Assertion failed for {description}: expected=[{string.Join(',', expected)}], "
                + $"actual=[{string.Join(',', actual)}]");
        }
    }

    private static void AssertFloatBitsEqual(float expected, float actual, string description)
    {
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
        {
            throw new InvalidOperationException(
                $"Assertion failed for {description}: expected={expected:R}, actual={actual:R}");
        }
    }

    private sealed class PlanFixture : IDisposable
    {
        private PlanFixture(string root, string planDirectory, string summarySha256)
        {
            Root = root;
            PlanDirectory = planDirectory;
            SummarySha256 = summarySha256;
        }

        public string Root { get; }
        public string PlanDirectory { get; }
        public string SummarySha256 { get; }

        public static PlanFixture Create(bool firstRecordGateFailure = false)
        {
            var root = Path.Combine(Path.GetTempPath(), $"recipient-derived-plan-test-{Guid.NewGuid():N}");
            var planDirectory = Path.Combine(root, "plan");
            var imagesDirectory = Path.Combine(root, "images");
            Directory.CreateDirectory(planDirectory);
            Directory.CreateDirectory(imagesDirectory);
            var filterContractPath = Path.GetFullPath(Path.Combine(root, "strict-filter.py"));
            var filterContractBytes = new UTF8Encoding(false).GetBytes("# frozen strict filter\n");
            File.WriteAllBytes(filterContractPath, filterContractBytes);
            var records = new List<string>(63);
            var inputs = new StringBuilder();
            for (var index = 0; index < 63; index++)
            {
                var source = Path.GetFullPath(Path.Combine(imagesDirectory, $"receipt-{index:000}.jpg"));
                var sourceBytes = Encoding.UTF8.GetBytes($"fixture-image-{index:000}");
                File.WriteAllBytes(source, sourceBytes);
                var sourceSha256 = RecipientDerivedCropHash.Sha256(sourceBytes);
                var gateFailure = firstRecordGateFailure && index == 0;
                var payload = new Dictionary<string, object?>
                {
                    ["schema_version"] = 1,
                    ["kind"] = RecipientDerivedCropShadowProgram.PlanRecordKind,
                    ["diagnostic_only"] = true,
                    ["formal_delivery_gate"] = false,
                    ["candidate_write_enabled"] = false,
                    ["source"] = source,
                    ["source_image"] = new Dictionary<string, object?>
                    {
                        ["path"] = source,
                        ["sha256"] = sourceSha256,
                        ["size_bytes"] = sourceBytes.LongLength,
                    },
                    ["rectification"] = RecipientDerivedCropShadowProgram.PlanRectification,
                    ["rectified_size"] = new Dictionary<string, object?>
                    {
                        ["width"] = 1000,
                        ["height"] = 1600,
                    },
                    ["detector_geometry"] = new Dictionary<string, object?>
                    {
                        ["amount_box"] = new[] { 200.0, 300.0, 800.0, 500.0 },
                        ["recipient_box"] = new[] { 100.0, 600.0, 900.0, 700.0 },
                        ["payment_box"] = new[] { 200.0, 800.0, 800.0, 900.0 },
                    },
                    ["global_gate_evidence"] = new Dictionary<string, object?>
                    {
                        ["recipient_detector_score"] = 0.95,
                        ["minimum_recipient_detector_score"] = 0.68,
                        ["ordinary_25pct_geometry_verified"] = !gateFailure,
                        ["alternative_envelope_verified"] = true,
                        ["global_gate_failures"] = gateFailure
                            ? new[] { "ordinary_25pct_geometry_not_verified" }
                            : Array.Empty<string>(),
                    },
                    ["existing_attempts"] = new Dictionary<string, object?>
                    {
                        ["first"] = Attempt(),
                        ["retry"] = Attempt(),
                        ["right_value"] = Attempt(),
                    },
                    ["crops"] = new object[]
                    {
                        Crop(
                            RecipientDerivedCropShadowProgram.Crop4,
                            [280, 565, 964, 735]),
                        Crop(
                            RecipientDerivedCropShadowProgram.Crop5,
                            [360, 592, 964, 708]),
                    },
                };
                using (var canonicalDocument = JsonDocument.Parse(JsonSerializer.Serialize(payload)))
                {
                    payload["plan_id"] = RecipientDerivedCropPlanContract.CanonicalPlanId(
                        canonicalDocument.RootElement);
                }
                records.Add(JsonSerializer.Serialize(payload));
                inputs.Append(source).Append('\n');
            }
            var plansBytes = new UTF8Encoding(false).GetBytes(string.Join("\n", records) + "\n");
            var inputsBytes = new UTF8Encoding(false).GetBytes(inputs.ToString());
            var plansPath = Path.Combine(planDirectory, "plans.jsonl");
            var inputsPath = Path.Combine(planDirectory, "inputs.txt");
            File.WriteAllBytes(plansPath, plansBytes);
            File.WriteAllBytes(inputsPath, inputsBytes);
            var summary = new Dictionary<string, object?>
            {
                ["schema_version"] = 1,
                ["kind"] = RecipientDerivedCropShadowProgram.PlanSummaryKind,
                ["diagnostic_only"] = true,
                ["formal_delivery_gate"] = false,
                ["candidate_write_enabled"] = false,
                ["ocr_rerun"] = false,
                ["production_output_changed"] = false,
                ["frozen_v4"] = new Dictionary<string, object?>
                {
                    ["formal_failures"] = 204,
                    ["candidate_records"] = 75,
                    ["remaining_records"] = 129,
                    ["remaining_with_global_gate_failures"] = 66,
                    ["remaining_with_clear_global_gates"] = 63,
                },
                ["records"] = 63,
                ["crop_names"] = new[]
                {
                    RecipientDerivedCropShadowProgram.Crop4,
                    RecipientDerivedCropShadowProgram.Crop5,
                },
                ["route_contract"] = new Dictionary<string, object?>
                {
                    ["crop4_requires_exact_match_with_existing_strict_crop"] = true,
                    ["crop5_requires_unique_exact_crop4_crop5_agreement"] = true,
                    ["minimum_line_confidence"] = 0.80,
                    ["minimum_recipient_detector_score"] = 0.68,
                    ["requires_ordinary_25pct_geometry"] = true,
                    ["requires_alternative_envelope"] = true,
                    ["candidate_write_enabled"] = false,
                },
                ["required_layout_producer"] = new Dictionary<string, object?>
                {
                    ["api"] = "PaddleOcrEngine.RecognizeLayoutDiagnostic",
                    ["execution_provider"] = "cpu",
                    ["rectification"] = RecipientDerivedCropShadowProgram.PlanRectification,
                    ["requires_raw_quad_crop_and_rectified_coordinates"] = true,
                    ["requires_verified_paddle_bundle_identity"] = true,
                    ["required_summary_kind"] = RecipientDerivedCropShadowProgram.SummaryKind,
                    ["required_record_kind"] = RecipientDerivedCropShadowProgram.RecordKind,
                },
                ["filter_contract"] = new Dictionary<string, object?>
                {
                    ["path"] = filterContractPath,
                    ["sha256"] = RecipientDerivedCropHash.Sha256(filterContractBytes),
                    ["size_bytes"] = filterContractBytes.LongLength,
                },
                ["artifacts"] = new Dictionary<string, object?>
                {
                    ["plans"] = Artifact("plans.jsonl", plansBytes),
                    ["inputs"] = Artifact("inputs.txt", inputsBytes),
                },
            };
            var summaryBytes = new UTF8Encoding(false).GetBytes(
                JsonSerializer.Serialize(summary) + Environment.NewLine);
            File.WriteAllBytes(Path.Combine(planDirectory, "summary.json"), summaryBytes);
            return new PlanFixture(
                root,
                planDirectory,
                RecipientDerivedCropHash.Sha256(summaryBytes));
        }

        public void Dispose()
        {
            if (Directory.Exists(Root))
            {
                Directory.Delete(Root, recursive: true);
            }
        }

        private static Dictionary<string, object?> Crop(string name, int[] box)
        {
            return new Dictionary<string, object?>
            {
                ["name"] = name,
                ["rectified_box"] = box,
                ["width"] = box[2] - box[0],
                ["height"] = box[3] - box[1],
                ["pixel_box_semantics"] = "left_top_inclusive_right_bottom_exclusive",
            };
        }

        private static Dictionary<string, object?> Attempt()
        {
            return new Dictionary<string, object?>
            {
                ["lines"] = Array.Empty<object>(),
            };
        }

        private static Dictionary<string, object?> Artifact(string path, byte[] bytes)
        {
            return new Dictionary<string, object?>
            {
                ["path"] = path,
                ["sha256"] = RecipientDerivedCropHash.Sha256(bytes),
                ["size_bytes"] = bytes.LongLength,
                ["records"] = 63,
            };
        }
    }
}
