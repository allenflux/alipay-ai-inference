using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Routes only the detected recipient row through the verified PP-OCR ONNX
/// bundle.  The architecture-v13 result remains authoritative for amount,
/// time, payment method and visible transfer-status text.
/// </summary>
internal static class PaddleRecipientHybrid
{
    /// <summary>
    /// Replace v13's recipient candidate with the strict, left-anchored value
    /// read by PP-OCR det + angle-cls + SVTR_LCNet rec.  An absent crop,
    /// unreadable row or missing left anchor removes the recipient candidate;
    /// it never falls back to the lower-accuracy v13 recipient branch.
    /// </summary>
    public static UnifiedOcrReadResult OverrideRecipient(
        Image<Rgb24> source,
        IReadOnlyList<DetectionResult> detections,
        PaddleOcrEngine paddleOcr,
        UnifiedOcrReadResult unified)
    {
        var candidates = unified.Candidates.ToDictionary(
            item => item.Key,
            item => item.Value,
            StringComparer.Ordinal);
        candidates.Remove("recipient_field");

        var detection = detections.FirstOrDefault(item =>
            string.Equals(item.Label, "recipient_field", StringComparison.Ordinal));
        if (detection is null)
        {
            return unified with { Candidates = candidates };
        }

        using var crop = UnifiedOcrImageOps.CropFieldWithMargin(source, detection.BboxImage);
        if (crop is null)
        {
            return unified with { Candidates = candidates };
        }

        var read = paddleOcr.Recognize(crop);
        var value = PaddleRecipientValueParser.Parse(read.Text);
        if (value is null)
        {
            return unified with { Candidates = candidates };
        }

        var confidence = read.Confidence is { } score && float.IsFinite(score)
            ? Math.Clamp(score, 0.0f, 1.0f)
            : 0.0f;
        candidates["recipient_field"] = new UnifiedOcrCandidate(
            value,
            confidence,
            value,
            confidence,
            null,
            null,
            unified.TextDeliveryValue);
        return unified with { Candidates = candidates };
    }

}
