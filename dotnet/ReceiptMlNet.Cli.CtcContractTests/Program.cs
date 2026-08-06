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
            Console.WriteLine("PASS: optimized CTC decoding is bit-exact and preserves validation/tie contracts.");
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
