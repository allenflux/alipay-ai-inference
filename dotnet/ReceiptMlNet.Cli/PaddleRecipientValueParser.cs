using System.Text;

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
    internal const string IndependentCropExactConsensusRoute =
        "independent_crop_exact_consensus";
    internal const string IndependentCropDominantThreeCropConsensusRoute =
        "independent_crop_dominant_three_crop_consensus";

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
    private static readonly System.Text.RegularExpressions.Regex ConsensusPureAmountPattern = new(
        @"^[\u00a5\uffe5$]?\s*(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?\s*(?:\u5143)?$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex ConsensusTimePattern = new(
        @"(?<!\d)\d{1,2}[:\uff1a]\d{2}(?::\d{2})?(?!\d)",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private static readonly System.Text.RegularExpressions.Regex ConsensusCurrencyCodePattern = new(
        @"(?:^|[^a-z])(?:cny|rmb|usd|hkd|eur|gbp|jpy)(?:$|[^a-z])",
        System.Text.RegularExpressions.RegexOptions.IgnoreCase
            | System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private const string ConsensusAllowedPunctuation = "*\uff0a()\uff08\uff09\u00b7\u2022&\uff06_-\u2014.\uff0e/";
    private static readonly string[] ConsensusNegativeTokens =
        [
            "\u91d1\u989d",
            "\u4ed8\u6b3e",
            "\u6536\u6b3e",
            "\u652f\u4ed8",
            "\u8f6c\u8d26",
            "\u8f6c\u5e10",
            "\u6210\u529f",
            "\u5931\u8d25",
            "\u5904\u7406\u4e2d",
            "\u5f85\u5904\u7406",
            "\u65f6\u95f4",
            "\u8ba2\u5355",
            "\u6d3b\u52a8",
            "\u4f18\u60e0",
            "\u5956\u52b1",
            "\u7ea2\u5305",
            "\u79ef\u5206",
            "\u5145\u503c",
            "\u5546\u54c1",
            "\u4ea4\u6613\u72b6\u6001",
            "\u4ea4\u6613\u5355\u53f7",
            "\u6d41\u6c34\u53f7",
            "\u94f6\u884c",
            "\u94f6\u884c\u5361",
            "\u50a8\u84c4\u5361",
            "\u4fe1\u7528\u5361",
            "\u501f\u8bb0\u5361",
            "\u94f6\u8054",
            "\u652f\u4ed8\u5b9d",
            "\u5fae\u4fe1",
            "\u4f59\u989d",
            "\u82b1\u5457",
            "\u5c3e\u53f7",
            "\u5408\u8ba1",
            "\u603b\u8ba1",
            "\u5b9e\u4ed8",
            "\u5e94\u4ed8",
            "\u4eba\u6c11\u5e01",
        ];
    // Frozen formal evidence exposed three OCR spellings of the pinyin row
    // label itself. Keep this deliberately exact: short/opaque ASCII payee
    // names remain eligible for independent-crop consensus.
    private static readonly string[] ConsensusRecipientLabelPinyinKeys =
        ["shoukuanfang", "shoukuanting", "shoukudnfang"];
    // Exact, whole-line normalized UI labels only. This is intentionally not
    // a substring or edit-distance filter, so opaque ASCII payee names remain
    // eligible unless the complete cleaned line is one of these controls.
    private static readonly string[] ConsensusAsciiUiLineKeys =
        [
            "amount",
            "amountdue",
            "balance",
            "bankcard",
            "creditcard",
            "debitcard",
            "discount",
            "failed",
            "failure",
            "order",
            "orderid",
            "ordernumber",
            "payee",
            "payment",
            "paymentfailed",
            "paymentfailure",
            "paymentmethod",
            "paymentprocessing",
            "paymentstatus",
            "paymentsuccess",
            "paymentsuccessful",
            "pending",
            "processing",
            "recipient",
            "recipientaccount",
            "recipientnumber",
            "status",
            "success",
            "successful",
            "time",
            "transactionfailed",
            "transactionfailure",
            "transactionid",
            "transactionnumber",
            "transactionprocessing",
            "transactionstatus",
            "transactionsuccess",
            "transactionsuccessful",
            "transfer",
            "transferfailed",
            "transferfailure",
            "transferprocessing",
            "transferstatus",
            "transfersuccess",
            "transfersuccessful",
        ];
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
    /// Fail-closed recovery after every anchored and calibrated route has
    /// failed.  It mirrors the frozen failure probe's strict runtime
    /// shadow: exactly one cleaned, line-level value must occur in at least
    /// two distinct deterministic crops, every participating occurrence must
    /// meet the 0.80 line floor, and both the ordinary 25-percent geometry and
    /// alternative envelope must already be verified.  The candidate is
    /// derived only from OCR evidence; truth, source paths and file names are
    /// not inputs.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult? ParseIndependentCropExactConsensus(
        IReadOnlyList<string>? firstRawLines,
        IReadOnlyList<float>? firstRawLineConfidences,
        IReadOnlyList<string>? retryRawLines,
        IReadOnlyList<float>? retryRawLineConfidences,
        IReadOnlyList<string>? rightValueRawLines,
        IReadOnlyList<float>? rightValueRawLineConfidences,
        float recipientDetectorScore,
        bool ordinaryGeometryVerified,
        bool alternativeEnvelopeVerified)
    {
        if (!TryCollectIndependentCropConsensusCandidates(
                firstRawLines,
                firstRawLineConfidences,
                retryRawLines,
                retryRawLineConfidences,
                rightValueRawLines,
                rightValueRawLineConfidences,
                recipientDetectorScore,
                ordinaryGeometryVerified,
                alternativeEnvelopeVerified,
                out var cropsByCandidate))
        {
            return null;
        }

        var eligible = cropsByCandidate
            .Where(item => item.Value.Count >= 2)
            .ToArray();
        if (eligible.Length != 1)
        {
            return null;
        }
        return new PaddleRecipientAlternativeParseResult(
            eligible[0].Key,
            IndependentCropExactConsensusRoute,
            CandidateConfidence: eligible[0].Value.Values.Min());
    }

    /// <summary>
    /// Last zero-inference fallback after ordinary two-crop exact consensus
    /// remains ambiguous.  Exactly one eligible value must occur in all three
    /// already-computed deterministic crops; every other eligible value may
    /// occur in at most two.  It deliberately shares the complete strict line
    /// and global-gate collector with the ordinary consensus route.
    /// </summary>
    public static PaddleRecipientAlternativeParseResult?
        ParseIndependentCropDominantThreeCropConsensus(
            IReadOnlyList<string>? firstRawLines,
            IReadOnlyList<float>? firstRawLineConfidences,
            IReadOnlyList<string>? retryRawLines,
            IReadOnlyList<float>? retryRawLineConfidences,
            IReadOnlyList<string>? rightValueRawLines,
            IReadOnlyList<float>? rightValueRawLineConfidences,
            float recipientDetectorScore,
            bool ordinaryGeometryVerified,
            bool alternativeEnvelopeVerified)
    {
        if (!TryCollectIndependentCropConsensusCandidates(
                firstRawLines,
                firstRawLineConfidences,
                retryRawLines,
                retryRawLineConfidences,
                rightValueRawLines,
                rightValueRawLineConfidences,
                recipientDetectorScore,
                ordinaryGeometryVerified,
                alternativeEnvelopeVerified,
                out var cropsByCandidate))
        {
            return null;
        }

        var eligible = cropsByCandidate
            .Where(item => item.Value.Count >= 2)
            .ToArray();
        if (eligible.Length < 2)
        {
            // A sole eligible candidate belongs to the earlier ordinary
            // exact-consensus route, never to this ambiguity fallback.
            return null;
        }
        var dominant = eligible
            .Where(item => item.Value.Count == 3)
            .ToArray();
        if (dominant.Length != 1)
        {
            return null;
        }
        return new PaddleRecipientAlternativeParseResult(
            dominant[0].Key,
            IndependentCropDominantThreeCropConsensusRoute,
            CandidateConfidence: dominant[0].Value.Values.Min());
    }

    private static bool TryCollectIndependentCropConsensusCandidates(
        IReadOnlyList<string>? firstRawLines,
        IReadOnlyList<float>? firstRawLineConfidences,
        IReadOnlyList<string>? retryRawLines,
        IReadOnlyList<float>? retryRawLineConfidences,
        IReadOnlyList<string>? rightValueRawLines,
        IReadOnlyList<float>? rightValueRawLineConfidences,
        float recipientDetectorScore,
        bool ordinaryGeometryVerified,
        bool alternativeEnvelopeVerified,
        out Dictionary<string, Dictionary<int, float>> cropsByCandidate)
    {
        cropsByCandidate = new Dictionary<string, Dictionary<int, float>>(
            StringComparer.Ordinal);
        if (!float.IsFinite(recipientDetectorScore)
            || recipientDetectorScore < 0.68f
            || recipientDetectorScore > 1.0f
            || !ordinaryGeometryVerified
            || !alternativeEnvelopeVerified)
        {
            return false;
        }

        var crops = new[]
        {
            (Lines: firstRawLines, Confidences: firstRawLineConfidences),
            (Lines: retryRawLines, Confidences: retryRawLineConfidences),
            (Lines: rightValueRawLines, Confidences: rightValueRawLineConfidences),
        };
        for (var cropIndex = 0; cropIndex < crops.Length; cropIndex++)
        {
            var (rawLines, rawConfidences) = crops[cropIndex];
            if (rawLines is null && rawConfidences is null)
            {
                continue;
            }
            if (rawLines is null
                || rawConfidences is null
                || rawLines.Count != rawConfidences.Count)
            {
                return false;
            }

            var bestByText = new Dictionary<string, float>(StringComparer.Ordinal);
            for (var lineIndex = 0; lineIndex < rawLines.Count; lineIndex++)
            {
                var confidence = rawConfidences[lineIndex];
                var text = ReceiptFieldNormalizer.CleanText(rawLines[lineIndex]);
                if (!float.IsFinite(confidence)
                    || confidence < 0.80f
                    || confidence > 1.0f
                    || !IsIndependentCropConsensusLineAllowed(text))
                {
                    continue;
                }
                if (!bestByText.TryGetValue(text, out var previous)
                    || confidence > previous)
                {
                    bestByText[text] = confidence;
                }
            }
            foreach (var (text, confidence) in bestByText)
            {
                if (!cropsByCandidate.TryGetValue(text, out var cropConfidences))
                {
                    cropConfidences = [];
                    cropsByCandidate[text] = cropConfidences;
                }
                cropConfidences[cropIndex] = confidence;
            }
        }
        return true;
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

    private static bool IsIndependentCropConsensusLineAllowed(string value)
    {
        if (value.Length == 0)
        {
            return false;
        }
        var visible = value.Replace(" ", string.Empty, StringComparison.Ordinal);
        var visibleRunes = visible.EnumerateRunes().ToArray();
        var normalizedAsciiLine = visible.ToLowerInvariant();
        if (visibleRunes.Length is < 2 or > 48
            || ConsensusNegativeTokens.Any(token =>
                visible.Contains(token, StringComparison.OrdinalIgnoreCase))
            || ConsensusRecipientLabelPinyinKeys.Contains(
                normalizedAsciiLine,
                StringComparer.Ordinal)
            || ConsensusAsciiUiLineKeys.Contains(
                normalizedAsciiLine,
                StringComparer.Ordinal)
            || visible.Contains('\u00a5')
            || visible.Contains('\uffe5')
            || ConsensusPureAmountPattern.IsMatch(value)
            || ConsensusCurrencyCodePattern.IsMatch(value)
            || ConsensusTimePattern.IsMatch(value))
        {
            return false;
        }

        var hasLetter = false;
        foreach (var rune in visibleRunes)
        {
            var category = System.Text.Rune.GetUnicodeCategory(rune);
            if (category is System.Globalization.UnicodeCategory.UppercaseLetter
                or System.Globalization.UnicodeCategory.LowercaseLetter
                or System.Globalization.UnicodeCategory.TitlecaseLetter
                or System.Globalization.UnicodeCategory.ModifierLetter
                or System.Globalization.UnicodeCategory.OtherLetter)
            {
                hasLetter = true;
                continue;
            }
            if (category is System.Globalization.UnicodeCategory.DecimalDigitNumber
                or System.Globalization.UnicodeCategory.LetterNumber
                or System.Globalization.UnicodeCategory.OtherNumber
                || ConsensusAllowedPunctuation.Contains(
                    rune.ToString(),
                    StringComparison.Ordinal))
            {
                continue;
            }
            return false;
        }
        return hasLetter;
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
