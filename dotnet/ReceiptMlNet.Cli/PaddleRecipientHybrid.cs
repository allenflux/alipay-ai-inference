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
    /// Replace v13's recipient candidate with a strict PP-OCR det + angle-cls
    /// + SVTR_LCNet value.  The ordinary route requires a left recipient
    /// label.  A known label-free layout is accepted only as an ordered
    /// merchant/CNY-amount pair inside the detector-selected recipient row.
    /// It never falls back to the lower-accuracy v13 recipient branch.
    /// An absent or ambiguous row removes the recipient candidate.  A failed
    /// standard crop gets one deterministic left-context retry through the
    /// same full PP-OCR pipeline and fail-closed parsers.
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
        var amountCandidate = unified.Candidates.TryGetValue("amount", out var amount)
            ? amount.Candidate
            : null;
        var verifiedUnlabelledLayout = HasVerifiedUnlabelledMerchantRowLayout(
            source, detections, detection);
        if (value is null
            && verifiedUnlabelledLayout
            && HasReliablePairLines(firstRead))
        {
            value = PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                firstRead.Lines.Select(line => line.Text).ToArray(),
                amountCandidate);
            if (value is not null)
            {
                route = "primary_unlabelled_merchant_amount_pair";
            }
        }
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
                var retryRoute = "left_context_retry";
                if (retryValue is null
                    && verifiedUnlabelledLayout
                    && HasReliablePairLines(retryRead))
                {
                    retryValue = PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                        retryRead.Lines.Select(line => line.Text).ToArray(),
                        amountCandidate);
                    retryRoute = "left_context_retry_unlabelled_merchant_amount_pair";
                }
                if (retryValue is not null)
                {
                    route = retryRoute;
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
                    : "anchored_or_pair_parse_failed"
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

    private static bool HasReliablePairLines(PaddleOcrReadResult read)
    {
        var nonEmptyLines = read.Lines
            .Where(line => ReceiptFieldNormalizer.CleanText(line.Text).Length > 0)
            .ToArray();
        return nonEmptyLines.Length == 2
            && nonEmptyLines.All(line => float.IsFinite(line.Confidence) && line.Confidence >= 0.80f);
    }

    private static bool HasVerifiedUnlabelledMerchantRowLayout(
        Image<Rgb24> source,
        IReadOnlyList<DetectionResult> detections,
        DetectionResult recipient)
    {
        var amount = detections.FirstOrDefault(item =>
            string.Equals(item.Label, "amount", StringComparison.Ordinal));
        var payment = detections.FirstOrDefault(item =>
            string.Equals(item.Label, "payment_method_field", StringComparison.Ordinal));
        if (amount is null || payment is null)
        {
            return false;
        }
        return PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
            source.Width,
            source.Height,
            recipient.Score,
            recipient.BboxImage,
            amount.Score,
            amount.BboxImage,
            payment.Score,
            payment.BboxImage);
    }

}
