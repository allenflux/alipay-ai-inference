/// <summary>
/// Decodes v13's visible transfer-status CTC text and derives the conservative
/// semantic code used for diagnostics. Delivery remains review-only; this type
/// does not promote a pseudo-label-derived candidate into a business value.
/// </summary>
internal static class UnifiedStatusTextDecoder
{
    public static UnifiedStatusTextRead Decode(
        ReadOnlySpan<float> values,
        int timeSteps,
        int classCount,
        IReadOnlyList<string> characters)
    {
        var decoded = UnifiedCtcDecoder.Decode(values, timeSteps, classCount, characters);
        return new UnifiedStatusTextRead(
            decoded.Text,
            ReceiptFieldNormalizer.NormalizeStatus(decoded.Text),
            decoded.Confidence);
    }
}

internal sealed record UnifiedStatusTextRead(
    string Text,
    string Normalized,
    float Confidence);
