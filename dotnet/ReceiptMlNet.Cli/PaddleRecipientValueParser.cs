/// <summary>
/// Strict extraction contract shared by the PP-OCR recipient route and its
/// package-free .NET contract test.
/// </summary>
internal sealed record PaddleRecipientAlternativeParseResult(
    string Value,
    string Route,
    long? AmountDeltaFen = null,
    float? CandidateConfidence = null);

internal static class PaddleRecipientValueParser
{
    internal const string PinyinAnnotatedThreeLineRoute = "pinyin_annotated_three_line";
    internal const string PinyinAnnotatedThreeLineStrongAnchorsRoute =
        "pinyin_annotated_three_line_strong_anchors";
    internal const string UnlabelledCjkAmountExactRoute = "unlabelled_cjk_amount_exact";
    internal const string UnlabelledMaskedCjkRightFullAgreementRoute =
        "unlabelled_masked_cjk_right_full_agreement";
    internal const string TruncatedRecipientLabelEmptyMaskThreeCropAgreementRoute =
        "truncated_recipient_label_empty_mask_three_crop_agreement";
    internal const string UnlabelledCjkDiscountArithmeticExactRoute =
        "unlabelled_cjk_discount_arithmetic_exact";

    private static readonly string[] RecipientLabels =
        ["\u6536\u6b3e\u65b9", "\u6536\u6b3e\u4eba", "\u6536\u6b3e\u8d26\u6237", "\u6536\u6b3e\u8d26\u53f7"];
    private static readonly char[] RowSeparators = [' ', ':', '\uff1a', '-', '\u2014'];
    private static readonly string[] NonRecipientRowLabels =
        [
            "\u4ed8\u6b3e\u65b9\u5f0f",
            "\u652f\u4ed8\u65b9\u5f0f",
            "\u4ea4\u6613\u65b9\u5f0f",
            "\u4ed8\u6b3e\u6e20\u9053",
            "\u8f6c\u8d26\u6210\u529f",
            "\u652f\u4ed8\u6210\u529f",
            "\u4ea4\u6613\u6210\u529f",
            "\u4ed8\u6b3e\u6210\u529f",
            "\u91d1\u989d",
            "\u65f6\u95f4",
            "\u8ba2\u5355\u53f7",
            "\u5546\u54c1",
            "\u4f18\u60e0",
            "\u6d3b\u52a8",
            "\u5145\u503c",
            "\u5956\u52b1",
            "\u7ea2\u5305",
            "\u79ef\u5206",
            "\u5e7f\u544a",
            "\u63a8\u8350",
        ];
    private static readonly System.Text.RegularExpressions.Regex ExplicitCnyAmountPattern = new(
        @"^[\u00a5\uffe5]\s*(?<amount>(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?)$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex ExpectedCnyAmountPattern = new(
        @"^(?:[\u00a5\uffe5]\s*)?(?<amount>(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?)$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex NumericMerchantPattern = new(
        @"^[0-9]{2,8}$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex MaskedCjkMerchantPattern = new(
        @"^(?<base>[\u3400-\u9fff]{2,12})(?:\((?<ascii>[\u3400-\u9fff*]{1,6})\)|（(?<fullwidth>[\u3400-\u9fff*]{1,6})）)$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex EmptyMaskCjkMerchantPattern = new(
        @"^[\u3400-\u9fff]{2,8}(?:\(\)|（）)$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex StrictCjkMiddleDotMerchantPattern = new(
        @"^[\u3400-\u9fff\u00b7]{2,16}$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex StrictDiscountGrossPattern = new(
        @"^[\u00a5\uffe5]\s*(?<amount>[0-9]+\.[0-9]{2})$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex DiscountFenPattern = new(
        @"^(?:0[1-9]|[1-9][0-9])$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly string[] StrongAnchorNonRecipientFragments =
        [
            "\u6536\u6b3e",
            "\u4ed8\u6b3e",
            "\u652f\u4ed8",
            "\u8f6c\u8d26",
            "\u4ea4\u6613",
            "\u6210\u529f",
            "\u5931\u8d25",
            "\u5904\u7406",
            "\u8be6\u60c5",
            "\u8d26\u53f7",
            "\u8d26\u6237",
            "\u91d1\u989d",
            "\u65f6\u95f4",
            "\u8ba2\u5355",
            "\u5546\u54c1",
            "\u4f18\u60e0",
            "\u6d3b\u52a8",
            "\u5145\u503c",
            "\u5956\u52b1",
            "\u7ea2\u5305",
            "\u79ef\u5206",
            "\u5e7f\u544a",
            "\u63a8\u8350",
            "\u9996\u9875",
            "\u8fd4\u56de",
            "\u9886\u53d6",
            "\u7acb\u51cf",
            "\u65b9\u5f0f",
            "\u6e20\u9053",
            "\u94f6\u884c",
            "\u50a8\u84c4\u5361",
            "\u4fe1\u7528\u5361",
        ];

    /// <summary>
    /// Match the acceptance evaluator exactly: the visible row must begin
    /// with a known recipient label and contain a non-empty right-side value.
    /// A label in the middle or a value before the label is rejected.
    /// </summary>
    public static string? Parse(string? rawText)
    {
        var text = ReceiptFieldNormalizer.CleanText(rawText);
        foreach (var label in RecipientLabels)
        {
            if (!text.StartsWith(label, StringComparison.Ordinal))
            {
                continue;
            }

            var value = text[label.Length..].TrimStart(RowSeparators);
            return value.Length == 0 ? null : value;
        }
        return null;
    }

    /// <summary>
    /// Accept the observed three-line pinyin annotation layout only when its
    /// detector and every OCR line pass their independent floors.  The
    /// normalised order is exactly "shou kuan fang", a CJK merchant value,
    /// then the Chinese recipient label.  The ordinary merchant floor stays
    /// at 0.70; a separate 0.67 tier requires independently strong detector,
    /// pinyin and exact Chinese-label anchors.  This does not relax the
    /// ordinary left-label parser above.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult? ParsePinyinAnnotatedRecipient(
        IReadOnlyList<string>? rawLines,
        IReadOnlyList<float>? rawLineConfidences,
        float recipientDetectorScore)
    {
        if (rawLines is null
            || rawLines.Count != 3
            || !float.IsFinite(recipientDetectorScore)
            || recipientDetectorScore < 0.90f
            || !TryPrepareLines(rawLines, rawLineConfidences, 3, out var lines))
        {
            return null;
        }

        var hasStrongPinyinAnchors = recipientDetectorScore >= 0.92f
            && lines[0].Confidence >= 0.94f
            && lines[2].Confidence >= 0.99f;
        var merchantConfidenceFloor = hasStrongPinyinAnchors ? 0.67f : 0.70f;
        if (lines[0].Confidence < 0.80f
            || lines[1].Confidence < merchantConfidenceFloor
            || lines[2].Confidence < 0.80f
            || !string.Equals(NormalizePinyin(lines[0].Text), "shoukuanfang", StringComparison.Ordinal)
            || !IsCjkMerchantCandidate(lines[1].Text)
            || !string.Equals(lines[2].Text, RecipientLabels[0], StringComparison.Ordinal))
        {
            return null;
        }
        return new PaddleRecipientAlternativeParseResult(
            lines[1].Text,
            hasStrongPinyinAnchors && lines[1].Confidence < 0.70f
                ? PinyinAnnotatedThreeLineStrongAnchorsRoute
                : PinyinAnnotatedThreeLineRoute);
    }

    /// <summary>
    /// Some receipt layouts omit a visible recipient label.  Accept exactly
    /// two ordered lines only: a merchant value followed by an explicit CNY
    /// amount.  CJK names use calibrated aggregate-score tiers based on the
    /// absolute amount difference in fen.  Numeric merchant identifiers have
    /// a separate, deliberately narrower exact-amount contract.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult? ParseUnlabelledMerchantAmountPair(
        IReadOnlyList<string>? rawLines,
        IReadOnlyList<float>? rawLineConfidences,
        string? expectedReceiptAmount,
        float? recipientDetectorScore)
    {
        if (rawLines is null
            || rawLines.Count != 2
            || !TryPrepareLines(rawLines, rawLineConfidences, 2, out var lines)
            || lines.Any(line => line.Confidence < 0.80f)
            || recipientDetectorScore is not { } score
            || !float.IsFinite(score)
            || !TryParseFullAmountFen(ExplicitCnyAmountPattern, lines[1].Text, out var observedFen)
            || !TryParseFullAmountFen(ExpectedCnyAmountPattern, expectedReceiptAmount, out var expectedFen))
        {
            return null;
        }

        var amountDeltaFen = observedFen >= expectedFen
            ? observedFen - expectedFen
            : expectedFen - observedFen;
        var merchant = lines[0].Text;
        if (IsCjkMerchantCandidate(merchant))
        {
            var requiredScore = amountDeltaFen switch
            {
                0 => 0.75f,
                <= 1 => 0.68f,
                <= 100 => 0.90f,
                _ => float.PositiveInfinity,
            };
            if (score < requiredScore)
            {
                return null;
            }
            var route = amountDeltaFen switch
            {
                0 => UnlabelledCjkAmountExactRoute,
                <= 1 => "unlabelled_cjk_amount_within_one_fen",
                _ => "unlabelled_cjk_amount_within_one_yuan",
            };
            return new PaddleRecipientAlternativeParseResult(merchant, route, amountDeltaFen);
        }

        if (!NumericMerchantPattern.IsMatch(merchant)
            || score < 0.95f
            || amountDeltaFen != 0
            || !long.TryParse(
                merchant,
                System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture,
                out var merchantNumber)
            || merchantNumber == expectedFen / 100L)
        {
            return null;
        }
        return new PaddleRecipientAlternativeParseResult(
            merchant,
            "unlabelled_numeric_amount_exact",
            amountDeltaFen);
    }

    /// <summary>
    /// Recover a masked label-free merchant only when the full-width retry and
    /// an independently narrow right-value crop contain the same single OCR
    /// line.  Parentheses must be paired in one style and the mask must contain
    /// at least one literal asterisk.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult? ParseUnlabelledMaskedCjkRightFullAgreement(
        IReadOnlyList<string>? fullRawLines,
        IReadOnlyList<float>? fullRawLineConfidences,
        int fullCropWidth,
        IReadOnlyList<string>? rightRawLines,
        IReadOnlyList<float>? rightRawLineConfidences,
        int rightCropWidth,
        int sourceWidth,
        float recipientDetectorScore)
    {
        if (!float.IsFinite(recipientDetectorScore)
            || recipientDetectorScore < 0.95f
            || fullRawLines is null
            || fullRawLines.Count != 1
            || rightRawLines is null
            || rightRawLines.Count != 1
            || !HasIndependentFullAndRightCrops(
                sourceWidth,
                fullCropWidth,
                rightCropWidth)
            || !TryPrepareLines(fullRawLines, fullRawLineConfidences, 1, out var fullLines)
            || !TryPrepareLines(rightRawLines, rightRawLineConfidences, 1, out var rightLines)
            || fullLines[0].Confidence < 0.80f
            || rightLines[0].Confidence < 0.80f
            || !string.Equals(fullLines[0].Text, rightLines[0].Text, StringComparison.Ordinal))
        {
            return null;
        }

        var value = fullLines[0].Text;
        var match = MaskedCjkMerchantPattern.Match(value);
        if (!match.Success || ContainsStrongAnchorNonRecipientFragment(value))
        {
            return null;
        }
        var masked = match.Groups["ascii"].Success
            ? match.Groups["ascii"].Value
            : match.Groups["fullwidth"].Value;
        if (!masked.Contains('*'))
        {
            return null;
        }

        return new PaddleRecipientAlternativeParseResult(
            value,
            UnlabelledMaskedCjkRightFullAgreementRoute,
            CandidateConfidence: Math.Min(fullLines[0].Confidence, rightLines[0].Confidence));
    }

    /// <summary>
    /// Recover the observed empty-mask layout only when the standard crop has
    /// the exact truncated label "\u6536\u6b3e" followed by a candidate and the
    /// full-width retry plus narrow right crop repeat that candidate exactly.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult? ParseTruncatedRecipientLabelEmptyMaskThreeCropAgreement(
        IReadOnlyList<string>? standardRawLines,
        IReadOnlyList<float>? standardRawLineConfidences,
        IReadOnlyList<string>? fullRawLines,
        IReadOnlyList<float>? fullRawLineConfidences,
        int fullCropWidth,
        IReadOnlyList<string>? rightRawLines,
        IReadOnlyList<float>? rightRawLineConfidences,
        int rightCropWidth,
        int sourceWidth,
        float recipientDetectorScore)
    {
        if (!float.IsFinite(recipientDetectorScore)
            || recipientDetectorScore < 0.95f
            || standardRawLines is null
            || standardRawLines.Count != 2
            || fullRawLines is null
            || fullRawLines.Count != 1
            || rightRawLines is null
            || rightRawLines.Count != 1
            || !HasIndependentFullAndRightCrops(sourceWidth, fullCropWidth, rightCropWidth)
            || !TryPrepareLines(standardRawLines, standardRawLineConfidences, 2, out var standardLines)
            || !TryPrepareLines(fullRawLines, fullRawLineConfidences, 1, out var fullLines)
            || !TryPrepareLines(rightRawLines, rightRawLineConfidences, 1, out var rightLines)
            || standardLines[0].Confidence < 0.50f
            || standardLines[1].Confidence < 0.80f
            || fullLines[0].Confidence < 0.75f
            || rightLines[0].Confidence < 0.75f
            || !string.Equals(standardLines[0].Text, "\u6536\u6b3e", StringComparison.Ordinal)
            || !string.Equals(standardLines[1].Text, fullLines[0].Text, StringComparison.Ordinal)
            || !string.Equals(fullLines[0].Text, rightLines[0].Text, StringComparison.Ordinal)
            || !EmptyMaskCjkMerchantPattern.IsMatch(standardLines[1].Text)
            || ContainsStrongAnchorNonRecipientFragment(standardLines[1].Text))
        {
            return null;
        }

        return new PaddleRecipientAlternativeParseResult(
            standardLines[1].Text,
            TruncatedRecipientLabelEmptyMaskThreeCropAgreementRoute,
            CandidateConfidence: new[]
            {
                standardLines[1].Confidence,
                fullLines[0].Confidence,
                rightLines[0].Confidence,
            }.Min());
    }

    /// <summary>
    /// Recover exactly one calibrated four-line merchant/price/discount/fen
    /// layout.  The visible original amount minus the unified receipt amount
    /// must equal the exactly two-digit discount in fen.  No other promotion
    /// marker is accepted.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult? ParseUnlabelledCjkDiscountArithmeticExact(
        IReadOnlyList<string>? rawLines,
        IReadOnlyList<float>? rawLineConfidences,
        string? expectedReceiptAmount,
        float recipientDetectorScore)
    {
        if (rawLines is null
            || rawLines.Count != 4
            || !float.IsFinite(recipientDetectorScore)
            || recipientDetectorScore < 0.95f
            || !TryPrepareLines(rawLines, rawLineConfidences, 4, out var lines)
            || lines[0].Confidence < 0.99f
            || lines[1].Confidence < 0.90f
            || lines[2].Confidence < 0.99f
            || lines[3].Confidence < 0.99f
            || !IsStrictCjkMiddleDotMerchant(lines[0].Text)
            || ContainsStrongAnchorNonRecipientFragment(lines[0].Text)
            || !TryParseFullAmountFen(StrictDiscountGrossPattern, lines[1].Text, out var originalFen)
            || !string.Equals(lines[2].Text, "\u767e\u6b21\u7acb\u51cf", StringComparison.Ordinal)
            || !DiscountFenPattern.IsMatch(lines[3].Text)
            || !long.TryParse(
                lines[3].Text,
                System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture,
                out var discountFen)
            || discountFen is < 1 or > 99
            || !TryParseFullAmountFen(ExpectedCnyAmountPattern, expectedReceiptAmount, out var expectedFen)
            || originalFen <= expectedFen
            || originalFen - expectedFen != discountFen)
        {
            return null;
        }

        return new PaddleRecipientAlternativeParseResult(
            lines[0].Text,
            UnlabelledCjkDiscountArithmeticExactRoute,
            discountFen,
            lines.Min(line => line.Confidence));
    }

    /// <summary>
    /// A detector overlap exception is eligible only for the strongest
    /// alternative route: a CJK merchant plus an explicit CNY amount exactly
    /// equal to the unified amount.  Ordinary geometry remains authoritative
    /// for every other route and for lower-scoring recipient detections.
    /// </summary>
    public static bool AllowsExactCjkPaymentOverlapException(
        PaddleRecipientAlternativeParseResult? alternative,
        float recipientDetectorScore)
    {
        return alternative is not null
            && string.Equals(
                alternative.Route,
                UnlabelledCjkAmountExactRoute,
                StringComparison.Ordinal)
            && float.IsFinite(recipientDetectorScore)
            && recipientDetectorScore >= 0.84f;
    }

    /// <summary>
    /// The lower-confidence pinyin merchant tier is diagnostic until two
    /// independently cropped PP-OCR reads agree on the cleaned merchant and
    /// both pass the ordinary row geometry.  Keeping this primitive-only
    /// makes the gate executable in the package-free contract tests.
    /// </summary>
    public static bool RequiresDualCropAgreement(
        PaddleRecipientAlternativeParseResult? alternative)
    {
        return alternative is not null
            && string.Equals(
                alternative.Route,
                PinyinAnnotatedThreeLineStrongAnchorsRoute,
                StringComparison.Ordinal);
    }

    public static bool HasRequiredDualCropAgreement(
        PaddleRecipientAlternativeParseResult? first,
        bool? firstGeometryAccepted,
        PaddleRecipientAlternativeParseResult? retry,
        bool? retryGeometryAccepted)
    {
        return firstGeometryAccepted is true
            && retryGeometryAccepted is true
            && IsPinyinAlternative(first)
            && IsPinyinAlternative(retry)
            && (RequiresDualCropAgreement(first) || RequiresDualCropAgreement(retry))
            && string.Equals(first!.Value, retry!.Value, StringComparison.Ordinal);
    }

    /// <summary>
    /// Primitive-only geometry contract so the package-free executable tests
    /// can prove that the unlabelled recipient row is physically between the
    /// primary amount and payment-method rows, not merely center-sorted.
    /// </summary>
    public static bool HasVerifiedUnlabelledMerchantRowGeometry(
        int sourceWidth,
        int sourceHeight,
        float recipientScore,
        float[] recipientBox,
        float amountScore,
        float[] amountBox,
        float paymentScore,
        float[] paymentBox,
        float paymentOverlapFraction = 0.25f)
    {
        if (sourceWidth < 2
            || sourceHeight < 2
            || !float.IsFinite(recipientScore)
            || recipientScore < 0.68f
            || !float.IsFinite(amountScore)
            || amountScore < 0.80f
            || !float.IsFinite(paymentScore)
            || paymentScore < 0.80f
            || !float.IsFinite(paymentOverlapFraction)
            || paymentOverlapFraction < 0.0f
            || paymentOverlapFraction > 0.45f
            || !IsFiniteBox(recipientBox)
            || !IsFiniteBox(amountBox)
            || !IsFiniteBox(paymentBox))
        {
            return false;
        }

        var recipientWidth = recipientBox[2] - recipientBox[0];
        var recipientHeight = recipientBox[3] - recipientBox[1];
        var recipientCenterY = (recipientBox[1] + recipientBox[3]) * 0.5f;
        var amountCenterY = (amountBox[1] + amountBox[3]) * 0.5f;
        var paymentCenterY = (paymentBox[1] + paymentBox[3]) * 0.5f;
        var amountVerticalTolerance = Math.Max(4.0f, recipientHeight * 0.25f);
        var paymentVerticalTolerance = Math.Max(
            4.0f,
            recipientHeight * paymentOverlapFraction);
        return recipientBox[0] <= sourceWidth * 0.20f
            && recipientBox[2] >= sourceWidth * 0.80f
            && recipientWidth >= sourceWidth * 0.60f
            && recipientHeight <= sourceHeight * 0.15f
            && amountCenterY < recipientCenterY
            && recipientCenterY < paymentCenterY
            && recipientBox[1] >= amountBox[3] - amountVerticalTolerance
            && recipientBox[3] <= paymentBox[1] + paymentVerticalTolerance;
    }

    private static bool IsCjkMerchantCandidate(string value)
    {
        if (value.Length is < 2 or > 64
            || value.Contains('\u00a5')
            || value.Contains('\uffe5')
            || RecipientLabels.Any(label => value.Contains(label, StringComparison.Ordinal))
            || NonRecipientRowLabels.Any(label => value.Contains(label, StringComparison.Ordinal)))
        {
            return false;
        }
        return value.Any(character => character is >= '\u3400' and <= '\u9fff');
    }

    private static bool ContainsStrongAnchorNonRecipientFragment(string value)
    {
        return StrongAnchorNonRecipientFragments.Any(fragment =>
            value.Contains(fragment, StringComparison.Ordinal));
    }

    private static bool IsStrictCjkMiddleDotMerchant(string value)
    {
        return StrictCjkMiddleDotMerchantPattern.IsMatch(value)
            && value[0] != '\u00b7'
            && value[^1] != '\u00b7'
            && !value.Contains("\u00b7\u00b7", StringComparison.Ordinal);
    }

    private static bool HasIndependentFullAndRightCrops(
        int sourceWidth,
        int fullCropWidth,
        int rightCropWidth)
    {
        return sourceWidth >= 2
            && fullCropWidth > 0
            && fullCropWidth <= sourceWidth
            && fullCropWidth >= sourceWidth * 0.90f
            && rightCropWidth > 0
            && rightCropWidth <= sourceWidth
            && rightCropWidth <= fullCropWidth * 0.65f;
    }

    private static bool IsPinyinAlternative(
        PaddleRecipientAlternativeParseResult? alternative)
    {
        return alternative is not null
            && (string.Equals(
                    alternative.Route,
                    PinyinAnnotatedThreeLineRoute,
                    StringComparison.Ordinal)
                || RequiresDualCropAgreement(alternative));
    }

    private static bool TryPrepareLines(
        IReadOnlyList<string>? rawLines,
        IReadOnlyList<float>? rawLineConfidences,
        int expectedCount,
        out PreparedLine[] lines)
    {
        if (!TryPrepareAnyLines(rawLines, rawLineConfidences, out lines))
        {
            return false;
        }
        return lines.Length == expectedCount;
    }

    private static bool TryPrepareAnyLines(
        IReadOnlyList<string>? rawLines,
        IReadOnlyList<float>? rawLineConfidences,
        out PreparedLine[] lines)
    {
        lines = [];
        if (rawLines is null
            || rawLineConfidences is null
            || rawLines.Count != rawLineConfidences.Count)
        {
            return false;
        }

        lines = rawLines
            .Select((text, index) => new PreparedLine(
                ReceiptFieldNormalizer.CleanText(text),
                rawLineConfidences[index]))
            .Where(line => line.Text.Length > 0)
            .ToArray();
        return lines.All(line => float.IsFinite(line.Confidence));
    }

    private static string NormalizePinyin(string value)
    {
        var decomposed = ReceiptFieldNormalizer.CleanText(value)
            .ToLowerInvariant()
            .Normalize(System.Text.NormalizationForm.FormD);
        var output = new System.Text.StringBuilder(decomposed.Length);
        foreach (var character in decomposed)
        {
            if (System.Globalization.CharUnicodeInfo.GetUnicodeCategory(character)
                == System.Globalization.UnicodeCategory.NonSpacingMark)
            {
                continue;
            }
            if (character is >= 'a' and <= 'z')
            {
                output.Append(character);
                continue;
            }
            if (!char.IsWhiteSpace(character))
            {
                return string.Empty;
            }
        }
        return output.ToString();
    }

    private static bool TryParseFullAmountFen(
        System.Text.RegularExpressions.Regex pattern,
        string? rawValue,
        out long amountFen)
    {
        amountFen = 0;
        var value = ReceiptFieldNormalizer.CleanText(rawValue);
        var match = pattern.Match(value);
        if (!match.Success)
        {
            return false;
        }
        var numeric = match.Groups["amount"].Value.Replace(",", string.Empty, StringComparison.Ordinal);
        if (!decimal.TryParse(
                numeric,
                System.Globalization.NumberStyles.AllowDecimalPoint,
                System.Globalization.CultureInfo.InvariantCulture,
                out var amount)
            || amount < 0m
            || amount > decimal.Truncate(long.MaxValue / 100m))
        {
            return false;
        }
        amountFen = decimal.ToInt64(
            decimal.Round(amount, 2, MidpointRounding.AwayFromZero) * 100m);
        return true;
    }

    private static bool IsFiniteBox(float[]? box)
    {
        return box is { Length: >= 4 }
            && box.Take(4).All(float.IsFinite)
            && box[2] > box[0]
            && box[3] > box[1];
    }

    private sealed record PreparedLine(string Text, float Confidence);
}
