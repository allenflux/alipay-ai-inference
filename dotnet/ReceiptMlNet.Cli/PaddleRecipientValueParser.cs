/// <summary>
/// Strict extraction contract shared by the PP-OCR recipient route and its
/// package-free .NET contract test.
/// </summary>
internal sealed record PaddleRecipientAlternativeParseResult(
    string Value,
    string Route,
    long? AmountDeltaFen = null);

internal static class PaddleRecipientValueParser
{
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
    /// then the Chinese recipient label.  This does not relax the ordinary
    /// left-label parser above.
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

        if (lines.Any(line => line.Confidence < 0.80f)
            || !string.Equals(NormalizePinyin(lines[0].Text), "shoukuanfang", StringComparison.Ordinal)
            || !IsCjkMerchantCandidate(lines[1].Text)
            || !string.Equals(lines[2].Text, RecipientLabels[0], StringComparison.Ordinal))
        {
            return null;
        }
        return new PaddleRecipientAlternativeParseResult(
            lines[1].Text,
            "pinyin_annotated_three_line");
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
                0 => "unlabelled_cjk_amount_exact",
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
        float[] paymentBox)
    {
        if (sourceWidth < 2
            || sourceHeight < 2
            || !float.IsFinite(recipientScore)
            || recipientScore < 0.68f
            || !float.IsFinite(amountScore)
            || amountScore < 0.80f
            || !float.IsFinite(paymentScore)
            || paymentScore < 0.80f
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
        var verticalTolerance = Math.Max(4.0f, recipientHeight * 0.25f);
        return recipientBox[0] <= sourceWidth * 0.20f
            && recipientBox[2] >= sourceWidth * 0.80f
            && recipientWidth >= sourceWidth * 0.60f
            && recipientHeight <= sourceHeight * 0.15f
            && amountCenterY < recipientCenterY
            && recipientCenterY < paymentCenterY
            && recipientBox[1] >= amountBox[3] - verticalTolerance
            && recipientBox[3] <= paymentBox[1] + verticalTolerance;
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

    private static bool TryPrepareLines(
        IReadOnlyList<string>? rawLines,
        IReadOnlyList<float>? rawLineConfidences,
        int expectedCount,
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
        return lines.Length == expectedCount
            && lines.All(line => float.IsFinite(line.Confidence));
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
