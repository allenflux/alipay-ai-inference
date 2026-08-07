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
    /// it never falls back to the lower-accuracy v13 recipient branch.  A
    /// failed standard crop gets one deterministic left-context retry, still
    /// through the full PP-OCR pipeline and the same strict parser.
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
            return unified with
            {
                Candidates = candidates,
                RecipientDiagnostic = new PaddleRecipientDiagnostic(
                    "none", "missing_detection", string.Empty, 0, 0, 0),
            };
        }

        using var crop = UnifiedOcrImageOps.CropFieldWithMargin(source, detection.BboxImage);
        if (crop is null)
        {
            return unified with
            {
                Candidates = candidates,
                RecipientDiagnostic = new PaddleRecipientDiagnostic(
                    "none", "invalid_standard_crop", string.Empty, 0, 0, 0),
            };
        }

        var firstRead = paddleOcr.Recognize(crop);
        var selectedRead = firstRead;
        var value = PaddleRecipientValueParser.Parse(firstRead.Text);
        string route = "primary";
        string? retryRaw = null;
        int? retryLineCount = null;
        int? retryCropWidth = null;
        int? retryCropHeight = null;

        if (value is null)
        {
            using var retryCrop = UnifiedOcrImageOps.CropRecipientRowLeftContext(source, detection.BboxImage);
            if (retryCrop is not null)
            {
                var retryRead = paddleOcr.Recognize(retryCrop);
                retryRaw = retryRead.Text;
                retryLineCount = retryRead.Lines.Count;
                retryCropWidth = retryCrop.Width;
                retryCropHeight = retryCrop.Height;
                var retryValue = PaddleRecipientValueParser.Parse(retryRead.Text);
                if (retryValue is not null)
                {
                    route = "left_context_retry";
                    selectedRead = retryRead;
                    value = retryValue;
                }
            }
        }

        var diagnostic = new PaddleRecipientDiagnostic(
            value is null ? "none" : route,
            value is null
                ? string.IsNullOrWhiteSpace(firstRead.Text) && string.IsNullOrWhiteSpace(retryRaw)
                    ? "ocr_empty"
                    : "anchor_parse_failed"
                : null,
            firstRead.Text,
            firstRead.Lines.Count,
            crop.Width,
            crop.Height,
            retryRaw,
            retryLineCount,
            retryCropWidth,
            retryCropHeight);
        if (value is null)
        {
            return unified with { Candidates = candidates, RecipientDiagnostic = diagnostic };
        }

        var confidence = selectedRead.Confidence is { } score && float.IsFinite(score)
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
        return unified with { Candidates = candidates, RecipientDiagnostic = diagnostic };
    }

}
