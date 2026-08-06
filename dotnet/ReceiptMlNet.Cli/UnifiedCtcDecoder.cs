internal static class UnifiedCtcDecoder
{
    /// <summary>
    /// Greedy CTC decoding with the same first-maximum tie policy and emitted
    /// confidence calculation as the original per-timestep ArgMax path.
    /// Every score is still scanned and checked for finiteness. The expensive
    /// softmax denominator is evaluated only when CTC emits a non-blank,
    /// non-repeated character because other timestep confidences are discarded.
    /// </summary>
    internal static (string Text, float Confidence) Decode(
        ReadOnlySpan<float> values,
        int timeSteps,
        int classCount,
        IReadOnlyList<string> characters)
    {
        if (classCount != characters.Count + 1)
        {
            throw new InvalidOperationException("Unified OCR CTC tensor differs from the verified character dictionary");
        }

        var text = new System.Text.StringBuilder();
        var scores = new List<float>();
        var previous = -1;
        for (var time = 0; time < timeSteps; time++)
        {
            var offset = checked(time * classCount);
            var maximumIndex = FindMaximumIndex(values, offset, classCount, out var maximum);
            if (maximumIndex != 0 && maximumIndex != previous)
            {
                text.Append(characters[maximumIndex - 1]);
                scores.Add(WinningSoftmaxConfidence(values, offset, classCount, maximum));
            }
            previous = maximumIndex;
        }
        return (text.ToString(), scores.Count == 0 ? 0.0f : scores.Average());
    }

    /// <summary>
    /// Returns the first maximum exactly as the old ArgMax implementation did.
    /// Finiteness validation intentionally remains unconditional so a blank or
    /// repeated timestep cannot hide a corrupt model output.
    /// </summary>
    internal static int FindMaximumIndex(ReadOnlySpan<float> values, int offset, int count, out float maximum)
    {
        if (count <= 0 || offset < 0 || offset + count > values.Length)
        {
            throw new InvalidOperationException("Unified OCR output contains an invalid score vector");
        }
        maximum = values[offset];
        if (!float.IsFinite(maximum))
        {
            throw new InvalidOperationException("Unified OCR output contains a non-finite score");
        }
        var maximumIndex = 0;
        for (var index = 1; index < count; index++)
        {
            var value = values[offset + index];
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("Unified OCR output contains a non-finite score");
            }
            if (value > maximum)
            {
                maximum = value;
                maximumIndex = index;
            }
        }
        return maximumIndex;
    }

    /// <summary>
    /// Computes the winning softmax probability in the original vocabulary
    /// order. Keeping both the order and double accumulator preserves the old
    /// floating-point confidence result for every emitted character.
    /// </summary>
    internal static float WinningSoftmaxConfidence(ReadOnlySpan<float> values, int offset, int count, float maximum)
    {
        var denominator = 0.0;
        for (var index = 0; index < count; index++)
        {
            denominator += Math.Exp(values[offset + index] - maximum);
        }
        if (!double.IsFinite(denominator) || denominator <= 0.0)
        {
            throw new InvalidOperationException("Unified OCR output has an invalid softmax denominator");
        }
        return (float)(1.0 / denominator);
    }
}
