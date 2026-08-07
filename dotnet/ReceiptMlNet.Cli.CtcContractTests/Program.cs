using System.Text.Json;

internal static class Program
{
    private static readonly string[] Characters = ["A", "B"];

    private static int Main()
    {
        try
        {
            VerifyBlankAndRepeatParity();
            VerifyFirstMaximumTieParity();
            VerifyAllBlankResult();
            VerifyNonFiniteStillFailsOnDiscardedTimesteps();
            VerifyVisibleStatusTextNormalization();
            VerifyResultCacheSemantics();
            Console.WriteLine(
                "PASS: optimized CTC decoding is bit-exact, preserves validation/tie contracts, "
                + "normalizes visible transfer-status text, and rejects stale result-cache semantics.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void VerifyBlankAndRepeatParity()
    {
        // Winners: blank, A, A(repeat), blank, A, B, B(repeat).
        // Only the three emitted characters may contribute confidence.
        var values = new[]
        {
            4.0f, 1.0f, 0.0f,
            0.0f, 4.0f, 1.0f,
            0.0f, 3.0f, 1.0f,
            5.0f, 2.0f, 1.0f,
            0.0f, 2.0f, 1.0f,
            0.0f, 1.0f, 5.0f,
            0.0f, 1.0f, 4.0f,
        };
        VerifyLegacyParity(values, 7, "AAB", "blank/repeat");
    }

    private static void VerifyFirstMaximumTieParity()
    {
        // Strict '>' means the first maximum wins. Winners are blank, A,
        // A(repeat), blank, A; therefore the collapsed result is AA.
        var values = new[]
        {
            2.0f, 2.0f, 1.0f,
            0.0f, 3.0f, 3.0f,
            0.0f, 4.0f, 4.0f,
            5.0f, 1.0f, 5.0f,
            0.0f, 2.0f, 2.0f,
        };
        VerifyLegacyParity(values, 5, "AA", "tie");
    }

    private static void VerifyAllBlankResult()
    {
        var values = new[]
        {
            3.0f, 1.0f, 0.0f,
            2.0f, 0.0f, 1.0f,
        };
        var actual = UnifiedCtcDecoder.Decode(values, 2, 3, Characters);
        AssertEqual(string.Empty, actual.Text, "all-blank text");
        AssertFloatBitsEqual(0.0f, actual.Confidence, "all-blank confidence");
    }

    private static void VerifyNonFiniteStillFailsOnDiscardedTimesteps()
    {
        // If NaN were ignored, each corrupt row would be discarded by CTC:
        // the first is blank and the second repeats A. Full-vector validation
        // must therefore remain before the emit decision.
        AssertNonFiniteFails(
            [3.0f, 1.0f, float.NaN],
            "blank timestep");
        AssertNonFiniteFails(
            [0.0f, 4.0f, 1.0f, 0.0f, 3.0f, float.PositiveInfinity],
            "repeated timestep",
            timeSteps: 2);
    }

    private static void AssertNonFiniteFails(float[] values, string label, int timeSteps = 1)
    {
        try
        {
            UnifiedCtcDecoder.Decode(values, timeSteps, 3, Characters);
        }
        catch (InvalidOperationException error) when (
            error.Message == "Unified OCR output contains a non-finite score")
        {
            return;
        }
        throw new InvalidOperationException($"{label}: non-finite score was not rejected with the verified error");
    }

    private static void VerifyVisibleStatusTextNormalization()
    {
        var characters = new[] { "转", "账", "成", "功", "处", "理", "中", "失", "败" };
        VerifyStatusText(characters, [1, 2, 3, 4], "转账成功", "success");
        VerifyStatusText(characters, [5, 6, 7], "处理中", "pending");
        VerifyStatusText(characters, [1, 2, 8, 9], "转账失败", "failed");
        VerifyStatusText(characters, [7], "中", "unknown");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("未转账成功"), "negated success");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("没有转账成功"), "explicitly negated success");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("未能支付成功"), "unable success");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("不是交易成功"), "not a success");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("转账成功与否"), "success uncertainty suffix");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("转账成功不了"), "success negation suffix");
        AssertEqual("unknown", ReceiptFieldNormalizer.NormalizeStatus("转账成功吗"), "success question suffix");
        AssertEqual(
            "unknown",
            ReceiptFieldNormalizer.NormalizeStatus("无法确认该笔款项已经转账成功"),
            "long-distance negated success");
    }

    private static void VerifyResultCacheSemantics()
    {
        AssertCacheCurrent(
            $"{{\"result_schema_version\":{ReceiptResultCacheContract.SchemaVersion},"
            + $"\"result_semantics_version\":\"{ReceiptResultCacheContract.SemanticsVersion}\","
            + "\"source\":\"fixture.jpg\"}",
            expected: true,
            "current semantics");
        AssertCacheCurrent("{\"source\":\"legacy.jpg\"}", expected: false, "legacy result without versions");
        AssertCacheCurrent(
            $"{{\"result_schema_version\":{ReceiptResultCacheContract.SchemaVersion}}}",
            expected: false,
            "missing semantics version");
        AssertCacheCurrent(
            $"{{\"result_semantics_version\":\"{ReceiptResultCacheContract.SemanticsVersion}\"}}",
            expected: false,
            "missing schema version");
        AssertCacheCurrent(
            $"{{\"result_schema_version\":{ReceiptResultCacheContract.SchemaVersion + 1},"
            + $"\"result_semantics_version\":\"{ReceiptResultCacheContract.SemanticsVersion}\"}}",
            expected: false,
            "future schema version");
        AssertCacheCurrent(
            $"{{\"result_schema_version\":{ReceiptResultCacheContract.SchemaVersion},"
            + "\"result_semantics_version\":\"legacy-status-logit-argmax\"}",
            expected: false,
            "legacy status-logit semantics");
        AssertCacheCurrent(
            $"{{\"result_schema_version\":\"{ReceiptResultCacheContract.SchemaVersion}\","
            + $"\"result_semantics_version\":\"{ReceiptResultCacheContract.SemanticsVersion}\"}}",
            expected: false,
            "string schema version");
    }

    private static void AssertCacheCurrent(string json, bool expected, string label)
    {
        using var document = JsonDocument.Parse(json);
        var actual = ReceiptResultCacheContract.IsCurrent(document.RootElement);
        if (actual != expected)
        {
            throw new InvalidOperationException(
                $"{label}: expected cache-current={expected}, got {actual}");
        }
    }

    private static void VerifyStatusText(
        IReadOnlyList<string> characters,
        IReadOnlyList<int> winners,
        string expectedText,
        string expectedNormalized)
    {
        var classCount = characters.Count + 1;
        var values = Enumerable.Repeat(-3.0f, checked(winners.Count * classCount)).ToArray();
        for (var time = 0; time < winners.Count; time++)
        {
            values[checked(time * classCount + winners[time])] = 3.0f;
        }
        var decoded = UnifiedStatusTextDecoder.Decode(values, winners.Count, classCount, characters);
        AssertEqual(expectedText, decoded.Text, $"{expectedText} status CTC text");
        AssertEqual(expectedNormalized, decoded.Normalized, $"{expectedText} normalized status");
    }

    private static void VerifyLegacyParity(float[] values, int timeSteps, string expectedText, string label)
    {
        var expected = LegacyDecode(values, timeSteps, 3, Characters);
        var actual = UnifiedCtcDecoder.Decode(values, timeSteps, 3, Characters);
        AssertEqual(expectedText, expected.Text, $"{label} legacy text fixture");
        AssertEqual(expected.Text, actual.Text, $"{label} text");
        AssertFloatBitsEqual(expected.Confidence, actual.Confidence, $"{label} confidence");
    }

    // Frozen implementation from UnifiedOcrEngine before the optimisation:
    // it computes a full softmax on every timestep, even when CTC discards it.
    private static (string Text, float Confidence) LegacyDecode(
        float[] values,
        int timeSteps,
        int classCount,
        IReadOnlyList<string> characters)
    {
        var text = new System.Text.StringBuilder();
        var scores = new List<float>();
        var previous = -1;
        for (var time = 0; time < timeSteps; time++)
        {
            var decoded = LegacyArgMax(values, checked(time * classCount), classCount);
            if (decoded.Index != 0 && decoded.Index != previous)
            {
                text.Append(characters[decoded.Index - 1]);
                scores.Add(decoded.Confidence);
            }
            previous = decoded.Index;
        }
        return (text.ToString(), scores.Count == 0 ? 0.0f : scores.Average());
    }

    private static (int Index, float Confidence) LegacyArgMax(float[] values, int offset, int count)
    {
        var maximum = values[offset];
        var maximumIndex = 0;
        for (var index = 1; index < count; index++)
        {
            var value = values[offset + index];
            if (value > maximum)
            {
                maximum = value;
                maximumIndex = index;
            }
        }
        var denominator = 0.0;
        for (var index = 0; index < count; index++)
        {
            denominator += Math.Exp(values[offset + index] - maximum);
        }
        return (maximumIndex, (float)(1.0 / denominator));
    }

    private static void AssertEqual(string expected, string actual, string label)
    {
        if (!string.Equals(expected, actual, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{label}: expected '{expected}', got '{actual}'");
        }
    }

    private static void AssertFloatBitsEqual(float expected, float actual, string label)
    {
        var expectedBits = BitConverter.SingleToInt32Bits(expected);
        var actualBits = BitConverter.SingleToInt32Bits(actual);
        if (expectedBits != actualBits)
        {
            throw new InvalidOperationException(
                $"{label}: expected {expected:R} (0x{expectedBits:X8}), got {actual:R} (0x{actualBits:X8})");
        }
    }
}
