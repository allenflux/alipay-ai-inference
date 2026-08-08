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
    /// label.  Known label-free layouts are accepted only through the
    /// calibrated pinyin-annotation or merchant/CNY-amount contracts inside
    /// the detector-selected recipient row.
    /// It never falls back to the lower-accuracy v13 recipient branch.
    /// An absent or ambiguous row removes the recipient candidate.  A failed
    /// standard crop gets one deterministic left-context retry through the
    /// same full PP-OCR pipeline and fail-closed parsers.  If both reads fail,
    /// a right-value crop is read for diagnostics only; that third read is
    /// never parsed and can never create a candidate.
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
        var verifiedAlternativeEnvelope = HasVerifiedUnlabelledMerchantRowLayout(
            source, detections, detection, paymentOverlapFraction: 0.45f);
        PaddleRecipientAlternativeParseResult? firstAlternative = null;
        bool? firstAlternativeGeometryAccepted = null;
        if (value is null
            && verifiedAlternativeEnvelope)
        {
            firstAlternative = ParseCalibratedAlternative(
                firstRead,
                detection.Score,
                amountCandidate);
            if (firstAlternative is not null)
            {
                firstAlternativeGeometryAccepted = HasVerifiedCalibratedAlternativeRowLayout(
                    source,
                    detections,
                    detection,
                    firstAlternative);
            }
            if (firstAlternative is not null
                && firstAlternativeGeometryAccepted is true
                && !PaddleRecipientValueParser.RequiresDualCropAgreement(firstAlternative))
            {
                value = firstAlternative.Value;
                route = $"primary_{firstAlternative.Route}";
            }
        }
        string? retryRaw = null;
        int? retryLineCount = null;
        int? retryCropWidth = null;
        int? retryCropHeight = null;
        PaddleOcrReadResult? retryReadEvidence = null;
        PaddleRecipientAlternativeParseResult? retryAlternative = null;
        bool? retryAlternativeGeometryAccepted = null;

        if (value is null)
        {
            using var retryCrop = UnifiedOcrImageOps.CropRecipientRowLeftContext(source, detection.BboxImage);
            if (retryCrop is not null)
            {
                var retryRead = paddleOcr.Recognize(retryCrop);
                retryReadEvidence = retryRead;
                retryRaw = retryRead.Text;
                retryLineCount = retryRead.Lines.Count;
                retryCropWidth = retryCrop.Width;
                retryCropHeight = retryCrop.Height;
                var retryValue = PaddleRecipientValueParser.Parse(retryRead.Text);
                var retryRoute = "left_context_retry";
                if (retryValue is null
                    && verifiedAlternativeEnvelope)
                {
                    retryAlternative = ParseCalibratedAlternative(
                        retryRead,
                        detection.Score,
                        amountCandidate);
                    if (retryAlternative is not null)
                    {
                        retryAlternativeGeometryAccepted = HasVerifiedCalibratedAlternativeRowLayout(
                            source,
                            detections,
                            detection,
                            retryAlternative);
                    }
                    if (retryAlternative is not null
                        && retryAlternativeGeometryAccepted is true)
                    {
                        var dualCropAgreement = PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                            firstAlternative,
                            firstAlternativeGeometryAccepted,
                            retryAlternative,
                            retryAlternativeGeometryAccepted);
                        if ((!PaddleRecipientValueParser.RequiresDualCropAgreement(firstAlternative)
                                && !PaddleRecipientValueParser.RequiresDualCropAgreement(retryAlternative))
                            || dualCropAgreement)
                        {
                            retryValue = retryAlternative.Value;
                            retryRoute = dualCropAgreement
                                ? $"dual_crop_{PaddleRecipientValueParser.PinyinAnnotatedThreeLineStrongAnchorsRoute}"
                                : $"left_context_retry_{retryAlternative.Route}";
                        }
                    }
                }
                if (retryValue is not null)
                {
                    route = retryRoute;
                    selectedRead = retryRead;
                    value = retryValue;
                }
            }
        }

        string? thirdRoute = null;
        string? rightValueRaw = null;
        int? rightValueLineCount = null;
        int? rightValueCropWidth = null;
        int? rightValueCropHeight = null;
        IReadOnlyList<float>? rightValueLineConfidences = null;
        PaddleOcrReadResult? rightValueReadEvidence = null;
        if (value is null)
        {
            using var rightValueCrop = UnifiedOcrImageOps.CropRecipientRowRightValue(
                source,
                detection.BboxImage);
            if (rightValueCrop is not null)
            {
                // Observability only: do not call any recipient parser and do
                // not assign selectedRead/value from this third read.
                rightValueReadEvidence = paddleOcr.Recognize(rightValueCrop);
                thirdRoute = "right_value";
                rightValueRaw = rightValueReadEvidence.Text;
                rightValueLineCount = rightValueReadEvidence.Lines.Count;
                rightValueCropWidth = rightValueCrop.Width;
                rightValueCropHeight = rightValueCrop.Height;
                rightValueLineConfidences = rightValueReadEvidence.Lines
                    .Select(line => float.IsFinite(line.Confidence)
                        ? MathF.Round(Math.Clamp(line.Confidence, 0.0f, 1.0f), 6)
                        : 0.0f)
                    .ToArray();
            }
        }

        var diagnostic = new PaddleRecipientDiagnostic(
            value is null ? "none" : route,
            value is null
                ? string.IsNullOrWhiteSpace(firstRead.Text)
                    && string.IsNullOrWhiteSpace(retryRaw)
                    && string.IsNullOrWhiteSpace(rightValueRaw)
                    ? "ocr_empty"
                    : BuildFailureReason(
                        verifiedAlternativeEnvelope,
                        firstRead,
                        firstAlternative,
                        firstAlternativeGeometryAccepted,
                        retryReadEvidence,
                        retryAlternative,
                        retryAlternativeGeometryAccepted,
                        rightValueReadEvidence)
                : null,
            firstRead.Text,
            firstRead.Lines.Count,
            crop.Width,
            crop.Height,
            retryRaw,
            retryLineCount,
            retryCropWidth,
            retryCropHeight,
            thirdRoute,
            rightValueRaw,
            rightValueLineCount,
            rightValueCropWidth,
            rightValueCropHeight,
            rightValueLineConfidences);
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

    private static string BuildFailureReason(
        bool verifiedAlternativeEnvelope,
        PaddleOcrReadResult firstRead,
        PaddleRecipientAlternativeParseResult? firstAlternative,
        bool? firstAlternativeGeometryAccepted,
        PaddleOcrReadResult? retryRead,
        PaddleRecipientAlternativeParseResult? retryAlternative,
        bool? retryAlternativeGeometryAccepted,
        PaddleOcrReadResult? rightValueRead)
    {
        return string.Join(
            ";",
            "anchored_or_alternative_parse_failed",
            $"alternative_envelope={verifiedAlternativeEnvelope}",
            $"first={DescribeReadEvidence(firstRead, firstAlternative, firstAlternativeGeometryAccepted)}",
            $"retry={DescribeReadEvidence(retryRead, retryAlternative, retryAlternativeGeometryAccepted)}",
            $"right_value={DescribeReadEvidence(rightValueRead, null, null)}");
    }

    private static string DescribeReadEvidence(
        PaddleOcrReadResult? read,
        PaddleRecipientAlternativeParseResult? alternative,
        bool? geometryAccepted)
    {
        if (read is null)
        {
            return "none";
        }
        var lines = string.Join(
            ",",
            read.Lines.Select((line, index) =>
                $"{index}:{line.Confidence.ToString("R", System.Globalization.CultureInfo.InvariantCulture)}:{line.Text}"));
        return string.Join(
            ",",
            $"line_count={read.Lines.Count}",
            $"alternative_route={alternative?.Route ?? "none"}",
            $"geometry={geometryAccepted?.ToString() ?? "not_evaluated"}",
            $"lines=[{lines}]");
    }

    private static PaddleRecipientAlternativeParseResult? ParseCalibratedAlternative(
        PaddleOcrReadResult read,
        float recipientDetectorScore,
        string? expectedReceiptAmount)
    {
        var texts = read.Lines.Select(line => line.Text).ToArray();
        var confidences = read.Lines.Select(line => line.Confidence).ToArray();
        return PaddleRecipientValueParser.ParsePinyinAnnotatedRecipient(
                texts,
                confidences,
                recipientDetectorScore)
            ?? PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                texts,
                confidences,
                expectedReceiptAmount,
                recipientDetectorScore);
    }

    private static bool HasVerifiedUnlabelledMerchantRowLayout(
        Image<Rgb24> source,
        IReadOnlyList<DetectionResult> detections,
        DetectionResult recipient,
        float paymentOverlapFraction = 0.25f)
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
            payment.BboxImage,
            paymentOverlapFraction);
    }

    private static bool HasVerifiedCalibratedAlternativeRowLayout(
        Image<Rgb24> source,
        IReadOnlyList<DetectionResult> detections,
        DetectionResult recipient,
        PaddleRecipientAlternativeParseResult alternative)
    {
        if (HasVerifiedUnlabelledMerchantRowLayout(source, detections, recipient))
        {
            return true;
        }
        return PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                alternative,
                recipient.Score)
            && HasVerifiedUnlabelledMerchantRowLayout(
                source,
                detections,
                recipient,
                paymentOverlapFraction: 0.45f);
    }

}
