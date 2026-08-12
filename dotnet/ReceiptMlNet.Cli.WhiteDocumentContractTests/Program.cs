using System.Security.Cryptography;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using OpenCvSharp;

internal static class Program
{
    private static int Main()
    {
        try
        {
            VerifyDocumentRoutingContract();
            VerifyWhiteDocumentOutputContract();
            VerifyWhiteRuntimeByteClosure();
            VerifyWhiteStudentBundleContract();
            VerifyWhiteStudentPreprocessContract();
            VerifyWhiteOutputFreshnessContract();
            Console.WriteLine(
                "PASS: blue/white routing, review-only Paddle/student JSON, shared crop ordering, and immutable device/PP-OCR/student byte-closure contracts.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void VerifyWhiteOutputFreshnessContract()
    {
        var root = Path.Combine(Path.GetTempPath(), $"white-output-freshness-{Guid.NewGuid():N}");
        ReceiptMlNetProgram.RequireFreshWhiteOutput(root);
        Directory.CreateDirectory(root);
        try
        {
            ExpectUsage(
                () => ReceiptMlNetProgram.RequireFreshWhiteOutput(root),
                "brand-new output directory");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void VerifyDocumentRoutingContract()
    {
        var legacy = CliOptions.Parse([
            "--detector", "receipt.onnx",
            "--input", "receipt.png",
            "--output", "output",
        ]);
        Assert(legacy.DocumentType == DocumentRoutePolicy.Blue, "legacy CLI must remain on the blue route");
        Assert(legacy.DetectorPath == "receipt.onnx", "legacy blue detector option changed");
        Assert(legacy.WhiteStudentBundlePath is null, "legacy blue route must not activate the white student");

        var white = CliOptions.Parse([
            "--document-type", "white",
            "--device-model", "device.onnx",
            "--ocr", "onnx",
            "--ocr-bundle", "bundle",
            "--white-student-bundle", "student",
            "--input", "white.png",
            "--output", "output",
            "--device", "cpu",
        ]);
        Assert(white.DocumentType == DocumentRoutePolicy.White, "white document route was not parsed");
        Assert(white.DetectorPath is null, "white route must not require a blue receipt detector");
        Assert(white.AnnotationMode == "none", "white route must default to JSON-only evidence");
        Assert(white.WhiteStudentBundlePath == "student", "white student bundle was not parsed");
        DocumentRoutePolicy.RequireRunnable(white.DocumentType);

        var auto = CliOptions.Parse([
            "--document-type", "auto",
            "--input", "unknown.png",
            "--output", "output",
        ]);
        ExpectUsage(
            () => DocumentRoutePolicy.RequireRunnable(auto.DocumentType),
            "no calibrated blue/white router");
        ExpectUsage(
            () => CliOptions.Parse([
                "--document-type", "white",
                "--device-model", "device.onnx",
                "--ocr", "onnx",
                "--ocr-bundle", "bundle",
                "--input", "white.png",
                "--output", "output",
                "--device", "auto",
            ]),
            "requires --device cpu");
        ExpectUsage(
            () => CliOptions.Parse([
                "--document-type", "white",
                "--device-model", "device.onnx",
                "--ocr", "onnx",
                "--ocr-bundle", "bundle",
                "--input", "white.png",
                "--output", "output",
                "--device", "cpu",
                "--annotate", "all",
            ]),
            "use --annotate none");
        ExpectUsage(
            () => CliOptions.Parse([
                "--detector", "receipt.onnx",
                "--input", "receipt.png",
                "--output", "output",
                "--white-student-bundle", "student",
            ]),
            "requires --document-type white");
    }

    private static void VerifyWhiteDocumentOutputContract()
    {
        var accepted = new PaddleOcrLine("到账成功", 0.91234565f);
        var rejected = new PaddleOcrLine("低置信", 0.12345678f);
        var read = new PaddleOcrLayoutReadResult(
            "到账成功",
            accepted.Confidence,
            new[] { accepted },
            new[]
            {
                new PaddleOcrLayoutLine(
                    accepted.Text,
                    accepted.Confidence,
                    Quad(1.12345f, 2.23456f),
                    true,
                    new WhiteLineStudentRead("到账成功", 0.81234565f)),
                new PaddleOcrLayoutLine(
                    rejected.Text,
                    rejected.Confidence,
                    Quad(11.0f, 12.0f),
                    false),
            });
        var lines = WhiteDocumentOutputContract.ProjectLines(read);
        AssertEqual(2, lines.Count, "white OCR line count");
        Assert(lines[0].PassesDropScore, "accepted white OCR line lost its threshold state");
        Assert(!lines[1].PassesDropScore, "below-threshold white OCR line must remain diagnostic only");
        Assert(lines[0].Quad.Count == 4, "white OCR quadrilateral must retain four points");
        Assert(lines[0].Student is not null, "white student comparison was not projected");
        Assert(
            lines[0].Student!.NormalizedExactMatch,
            "identical Paddle/student text must be marked normalized-exact");
        Assert(
            lines[0].Student!.CropSource == WhiteDocumentOutputContract.StudentCropSource,
            "white student crop source contract changed");
        Assert(
            WhiteDocumentOutputContract.NormalizedExactMatch("Ａ\tB", "A B"),
            "white Paddle/student comparison must use NFKC and collapsed whitespace");

        var assembled = PaddleOcrEngine.AssembleLayoutDiagnostic(
            [
                [new Point2f(0, 0), new Point2f(4, 0), new Point2f(4, 2), new Point2f(0, 2)],
                [new Point2f(0, 3), new Point2f(4, 3), new Point2f(4, 5), new Point2f(0, 5)],
            ],
            [accepted, rejected],
            0.5f,
            [new WhiteLineStudentRead("student-0", 0.8f), new WhiteLineStudentRead("student-1", 0.7f)]);
        Assert(
            assembled.Lines[0].Student?.Text == "student-0"
                && assembled.Lines[1].Student?.Text == "student-1",
            "student reads must remain bound to DB/CLS/Paddle REC line order");
        ExpectInvalidOperation(
            () => PaddleOcrEngine.AssembleLayoutDiagnostic(
                [[new Point2f(0, 0), new Point2f(4, 0), new Point2f(4, 2), new Point2f(0, 2)]],
                [accepted],
                0.5f,
                Array.Empty<WhiteLineStudentRead?>()),
            "box/student count differs");

        var result = new WhiteDocumentResult(
            WhiteDocumentOutputContract.SchemaVersion,
            WhiteDocumentOutputContract.SemanticsVersion,
            DocumentRoutePolicy.White,
            "white.png",
            "dotnet_onnxruntime_cpu",
            new ImageSize(100, 200),
            new WhiteDocumentRouteEvidence("white", "white", "explicit_cli", false, true, "not_calibrated"),
            new WhiteDocumentOcrEvidence(
                "paddle_teacher_candidate",
                "ppocr_db_cls_rec",
                "cpu",
                WhiteDocumentOutputContract.DeliveryPolicy,
                "integrated_review_only",
                "not_calibrated",
                read.Text,
                read.Confidence,
                1,
                2,
                "cpu",
                1,
                1,
                WhiteDocumentOutputContract.StudentCropSource),
            lines,
            new DeviceResult("ios", "苹果", "cnn", 0.9f, false, 0.9f, null, null),
            new WhiteDocumentContractReferences(
                "device.onnx", "d", "dc", "bundle.json", "bc", "ac",
                "det.onnx", "det", "cls.onnx", "cls", "rec.onnx", "rec", "dict.txt", "dict",
                "immutable_verified_bytes", false, 2, "dict"),
            new WhiteDocumentStageLatency(1, 2, 3, 4),
            ["review only"]);
        var json = JsonSerializer.Serialize(result, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        });
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        Assert(root.GetProperty("document_type").GetString() == "white", "white JSON document type missing");
        Assert(!root.TryGetProperty("fields", out _), "white JSON must not fabricate receipt fields");
        Assert(!root.TryGetProperty("detections", out _), "white JSON must not fabricate blue detections");
        Assert(root.GetProperty("route").GetProperty("review_required").GetBoolean(), "white JSON must require review");
        Assert(
            root.GetProperty("ocr").GetProperty("delivery_policy").GetString() == "review_only",
            "white OCR delivery policy must fail closed");
        Assert(
            root.GetProperty("model_contracts").GetProperty("runtime_source").GetString()
                == "immutable_verified_bytes",
            "white runtime must identify its byte-closed model source");
        Assert(
            !root.GetProperty("model_contracts").GetProperty("reopened_paths_after_verification").GetBoolean(),
            "white runtime must not reopen model paths after verification");
        AssertEqual(2, root.GetProperty("lines").GetArrayLength(), "white JSON line evidence count");
        Assert(
            root.GetProperty("lines")[0].GetProperty("passes_drop_score").GetBoolean(),
            "white JSON must expose threshold state without implying delivery");
        Assert(
            root.GetProperty("lines")[0].GetProperty("student").GetProperty("delivery_policy").GetString()
                == "review_only",
            "white student comparison must remain review-only");
        Assert(
            root.GetProperty("lines")[0].GetProperty("student").GetProperty("crop_source").GetString()
                == WhiteDocumentOutputContract.StudentCropSource,
            "white JSON must bind student to the shared DB/CLS crop");
        AssertEqual(
            4,
            root.GetProperty("lines")[0].GetProperty("quad").GetArrayLength(),
            "white JSON quadrilateral point count");
    }

    private static void VerifyWhiteRuntimeByteClosure()
    {
        var root = Path.Combine(Path.GetTempPath(), $"white-runtime-closure-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var modelPath = Path.Combine(root, "statusbar_device_v1.onnx");
            var contractPath = Path.ChangeExtension(modelPath, ".contract.json");
            var modelBytes = Encoding.UTF8.GetBytes("closed-device-model-fixture");
            var modelHash = Sha256(modelBytes);
            File.WriteAllBytes(modelPath, modelBytes);
            File.WriteAllText(
                contractPath,
                JsonSerializer.Serialize(new
                {
                    kind = "statusbar_device_v1",
                    onnx = new { sha256 = modelHash },
                }));
            var expectedContractHash = Sha256(File.ReadAllBytes(contractPath));
            var deviceSnapshot = DeviceModelCpuSnapshot.LoadAndVerify(modelPath);
            var contractHash = deviceSnapshot.Contract.ContractSha256;
            Assert(
                contractHash == expectedContractHash,
                "device contract identity was not computed from its parsed bytes");

            File.WriteAllBytes(modelPath, Encoding.UTF8.GetBytes("attacker-device-model-swap"));
            File.WriteAllText(contractPath, "{\"kind\":\"attacker\"}");
            Assert(
                deviceSnapshot.ClosedModelSha256 == modelHash,
                "device model snapshot changed after path replacement");
            Assert(
                deviceSnapshot.Contract.ContractSha256 == contractHash,
                "device contract identity changed after path replacement");

            var attackerModelBytes = modelBytes.ToArray();
            attackerModelBytes[0] ^= 0x01;
            ExpectInvalidOperation(
                () => DeviceModelCpuSnapshot.VerifyAndCloneModelBytes(attackerModelBytes, modelHash),
                "changed before the private CPU snapshot");

            var paddleModelBytes = Encoding.UTF8.GetBytes("closed-paddle-model-fixture");
            var paddleRecord = new PaddleOcrFileRecord(
                "det.onnx",
                Path.Combine(root, "det.onnx"),
                Sha256(paddleModelBytes),
                paddleModelBytes.LongLength);
            var paddleClone = PaddleOcrCpuModelSnapshot.VerifyAndClone(
                paddleModelBytes,
                paddleRecord,
                "detector fixture");
            paddleModelBytes[0] ^= 0x01;
            Assert(
                paddleClone[0] != paddleModelBytes[0],
                "Paddle model snapshot retained caller-owned mutable bytes");
            ExpectInvalidOperation(
                () => PaddleOcrCpuModelSnapshot.VerifyAndClone(
                    paddleModelBytes,
                    paddleRecord,
                    "detector fixture"),
                "differs from the verified delivery contract");

            var dictionaryBytes = new UTF8Encoding(false).GetBytes("商\n户\n");
            var dictionaryRecord = new PaddleOcrFileRecord(
                "dict.txt",
                Path.Combine(root, "dict.txt"),
                Sha256(dictionaryBytes),
                dictionaryBytes.LongLength);
            var dictionaryClone = PaddleOcrCpuRuntimeSnapshot.VerifyDictionaryAndClone(
                dictionaryBytes,
                dictionaryRecord,
                useSpaceCharacter: true,
                expectedRecognitionCharacters: ["商", "户", " "],
                expectedCtcCharacters: ["blank", "商", "户", " "]);
            dictionaryBytes[0] ^= 0x01;
            Assert(
                dictionaryClone[0] != dictionaryBytes[0],
                "Paddle dictionary snapshot retained caller-owned mutable bytes");
            ExpectInvalidOperation(
                () => PaddleOcrCpuRuntimeSnapshot.VerifyDictionaryAndClone(
                    dictionaryBytes,
                    dictionaryRecord,
                    useSpaceCharacter: true,
                    expectedRecognitionCharacters: ["商", "户", " "],
                    expectedCtcCharacters: ["blank", "商", "户", " "]),
                "differs from the verified delivery contract");

            var alternateDictionary = new UTF8Encoding(false).GetBytes("商\n甲\n");
            var alternateRecord = dictionaryRecord with
            {
                Sha256 = Sha256(alternateDictionary),
                SizeBytes = alternateDictionary.LongLength,
            };
            ExpectInvalidOperation(
                () => PaddleOcrCpuRuntimeSnapshot.VerifyDictionaryAndClone(
                    alternateDictionary,
                    alternateRecord,
                    useSpaceCharacter: true,
                    expectedRecognitionCharacters: ["商", "户", " "],
                    expectedCtcCharacters: ["blank", "商", "户", " "]),
                "differs from the verified in-memory vocabulary");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void VerifyWhiteStudentBundleContract()
    {
        var root = Path.Combine(Path.GetTempPath(), $"white-student-closure-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var modelBytes = Encoding.UTF8.GetBytes("closed-white-student-onnx-fixture");
            var charset = new
            {
                schema_version = 1,
                blank_index = 0,
                characters = new[] { "到", "账", "A" },
                sha256 = Sha256(Encoding.UTF8.GetBytes("到账A")),
            };
            var charsetBytes = new UTF8Encoding(false).GetBytes(JsonSerializer.Serialize(charset));
            var modelPath = Path.Combine(root, "generic_text_line.onnx");
            var charsetPath = Path.Combine(root, "generic_text_line.charset.json");
            var contractPath = Path.Combine(root, "generic_text_line.contract.json");
            File.WriteAllBytes(modelPath, modelBytes);
            File.WriteAllBytes(charsetPath, charsetBytes);
            File.WriteAllText(
                contractPath,
                JsonSerializer.Serialize(new
                {
                    schema_version = 1,
                    kind = WhiteLineStudentBundle.ContractKind,
                    onnx_file = Path.GetFileName(modelPath),
                    onnx_sha256 = Sha256(modelBytes),
                    charset_file = Path.GetFileName(charsetPath),
                    charset_sha256 = Sha256(charsetBytes),
                    fields = new[] { WhiteLineStudentBundle.FieldKind },
                    input = new
                    {
                        name = WhiteLineStudentBundle.InputName,
                        dtype = "float32",
                        shape = new[] { 1, 1, 32, 160 },
                        preprocess = WhiteLineStudentBundle.Preprocess,
                    },
                    output = new
                    {
                        name = WhiteLineStudentBundle.OutputName,
                        shape = new[] { 40, 1, 4 },
                        layout = "[time,batch,class]",
                        decoder = "ctc_greedy",
                        blank_index = 0,
                    },
                }));
            var bundle = WhiteLineStudentBundle.LoadAndVerify(root);
            var modelHash = bundle.ModelSha256;
            var charsetHash = bundle.CharsetSha256;
            var contractHash = bundle.ContractSha256;
            File.WriteAllText(modelPath, "attacker-model");
            File.WriteAllText(charsetPath, "attacker-charset");
            File.WriteAllText(contractPath, "attacker-contract");
            Assert(bundle.ModelSha256 == modelHash, "student model snapshot reopened its source path");
            Assert(bundle.CharsetSha256 == charsetHash, "student charset snapshot reopened its source path");
            Assert(bundle.ContractSha256 == contractHash, "student contract snapshot reopened its source path");
            Assert(bundle.ImageHeight == 32 && bundle.ImageWidth == 160, "student input shape changed");
            Assert(bundle.Characters.SequenceEqual(["到", "账", "A"]), "student charset ordering changed");

            var changed = modelBytes.ToArray();
            changed[0] ^= 0x01;
            ExpectInvalidOperation(
                () => WhiteLineStudentBundle.VerifyAndClone(
                    changed,
                    modelBytes.LongLength,
                    modelHash,
                    "model fixture"),
                "differs from its verified contract");

            File.WriteAllBytes(modelPath, modelBytes);
            File.WriteAllBytes(charsetPath, charsetBytes);
            File.WriteAllText(
                contractPath,
                JsonSerializer.Serialize(new
                {
                    schema_version = 1,
                    kind = WhiteLineStudentBundle.ContractKind,
                    onnx_file = Path.GetFileName(modelPath),
                    onnx_sha256 = Sha256(modelBytes),
                    charset_file = Path.GetFileName(charsetPath),
                    charset_sha256 = Sha256(charsetBytes),
                    fields = new[] { "amount" },
                    input = new { name = "image", dtype = "float32", shape = new[] { 1, 1, 32, 160 }, preprocess = WhiteLineStudentBundle.Preprocess },
                    output = new { name = "logits", shape = new[] { 40, 1, 4 }, layout = "[time,batch,class]", decoder = "ctc_greedy", blank_index = 0 },
                }));
            ExpectUsage(
                () => WhiteLineStudentBundle.LoadAndVerify(root),
                "fields must equal [generic_text_line]");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void VerifyWhiteStudentPreprocessContract()
    {
        var pixels = new byte[3, 5, 3]
        {
            { { 0, 0, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 255 } },
            { { 12, 34, 56 }, { 78, 90, 123 }, { 200, 10, 30 }, { 4, 250, 128 }, { 33, 66, 99 } },
            { { 250, 128, 4 }, { 17, 222, 19 }, { 90, 45, 180 }, { 1, 2, 3 }, { 127, 127, 127 } },
        };
        var expectedGray = new byte[,]
        {
            { 0, 76, 150, 29, 255 },
            { 30, 90, 69, 163, 60 },
            { 150, 138, 74, 2, 127 },
        };
        for (var y = 0; y < 3; y++)
        {
            for (var x = 0; x < 5; x++)
            {
                Assert(
                    WhiteLineStudentEngine.RgbToGray(
                        pixels[y, x, 0],
                        pixels[y, x, 1],
                        pixels[y, x, 2]) == expectedGray[y, x],
                    $"generic text line RGB->gray parity changed at ({x},{y})");
            }
        }

        if (!OperatingSystem.IsWindows())
        {
            // The delivery project intentionally ships only the Windows x64
            // OpenCvSharp native runtime. The same test runs the full resize
            // and tensor hash on the supported delivery OS.
            return;
        }

        using var rgb = new Mat(3, 5, MatType.CV_8UC3);
        for (var y = 0; y < 3; y++)
        {
            for (var x = 0; x < 5; x++)
            {
                rgb.Set(
                    y,
                    x,
                    new Vec3b(pixels[y, x, 0], pixels[y, x, 1], pixels[y, x, 2]));
            }
        }
        var tensor = WhiteLineStudentEngine.PrepareInput(rgb, 7, 11);
        var tensorBytes = MemoryMarshal.AsBytes(tensor.AsSpan()).ToArray();
        // Independent Python reference: cv2.INTER_LINEAR_EXACT over the same
        // 3x5 RGB fixture, integer gray formula, 7x11 white canvas, float32 /255.
        Assert(
            Sha256(tensorBytes) == "ee1b0457871cc38344a994509057a99aebc96cc3820c678838240ecddf185c7f",
            "generic text line C#/Python final float32 NCHW parity hash changed");
    }

    private static IReadOnlyList<PaddleOcrLayoutPoint> Quad(float x, float y)
    {
        return new[]
        {
            new PaddleOcrLayoutPoint(x, y),
            new PaddleOcrLayoutPoint(x + 5, y),
            new PaddleOcrLayoutPoint(x + 5, y + 4),
            new PaddleOcrLayoutPoint(x, y + 4),
        };
    }

    private static string Sha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static void ExpectUsage(Action action, string expectedMessage)
    {
        try
        {
            action();
        }
        catch (UsageException exception) when (exception.Message.Contains(expectedMessage, StringComparison.Ordinal))
        {
            return;
        }
        throw new InvalidOperationException($"Expected UsageException containing: {expectedMessage}");
    }

    private static void ExpectInvalidOperation(Action action, string expectedMessage)
    {
        try
        {
            action();
        }
        catch (InvalidOperationException exception)
            when (exception.Message.Contains(expectedMessage, StringComparison.Ordinal))
        {
            return;
        }
        throw new InvalidOperationException(
            $"Expected InvalidOperationException containing: {expectedMessage}");
    }

    private static void AssertEqual(int expected, int actual, string label)
    {
        if (expected != actual)
        {
            throw new InvalidOperationException($"{label}: expected {expected}, actual {actual}");
        }
    }

    private static void Assert(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
