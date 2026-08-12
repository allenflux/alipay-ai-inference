using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

internal static class Program
{
    private static int Main()
    {
        try
        {
            VerifyDocumentRoutingContract();
            VerifyWhiteDocumentOutputContract();
            VerifyWhiteRuntimeByteClosure();
            Console.WriteLine(
                "PASS: blue/white routing, review-only JSON, and immutable device/PP-OCR byte-closure contracts.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
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

        var white = CliOptions.Parse([
            "--document-type", "white",
            "--device-model", "device.onnx",
            "--ocr", "onnx",
            "--ocr-bundle", "bundle",
            "--input", "white.png",
            "--output", "output",
            "--device", "cpu",
        ]);
        Assert(white.DocumentType == DocumentRoutePolicy.White, "white document route was not parsed");
        Assert(white.DetectorPath is null, "white route must not require a blue receipt detector");
        Assert(white.AnnotationMode == "none", "white route must default to JSON-only evidence");
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
                    true),
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
                "not_integrated",
                "not_calibrated",
                read.Text,
                read.Confidence,
                1,
                2),
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
