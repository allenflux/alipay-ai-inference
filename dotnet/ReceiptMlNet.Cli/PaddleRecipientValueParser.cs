/// <summary>
/// Strict extraction contract shared by the PP-OCR recipient route and its
/// package-free .NET contract test.
/// </summary>
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
    /// Some receipt layouts omit a visible recipient label.  Accept that
    /// variant only when PP-OCR detects exactly two ordered lines in the
    /// detector-selected recipient row: a CJK merchant name followed by an
    /// explicit CNY amount.  The amount is structural corroboration and is
    /// never returned as recipient text.  Any additional line, reversed
    /// order, missing currency mark, known non-recipient label, or non-CJK
    /// value remains fail-closed.
    /// </summary>
    public static string? ParseUnlabelledMerchantAmountPair(
        IReadOnlyList<string>? rawLines,
        string? expectedReceiptAmount)
    {
        if (rawLines is null)
        {
            return null;
        }

        var lines = rawLines
            .Select(ReceiptFieldNormalizer.CleanText)
            .Where(line => line.Length > 0)
            .ToArray();
        if (lines.Length != 2
            || !IsMerchantCandidate(lines[0])
            || !IsExplicitCnyAmount(lines[1], expectedReceiptAmount))
        {
            return null;
        }
        return lines[0];
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
            || recipientScore < 0.90f
            || amountScore < 0.80f
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

    private static bool IsMerchantCandidate(string value)
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

    private static bool IsExplicitCnyAmount(string value, string? expectedReceiptAmount)
    {
        return TryParseFullAmountFen(ExplicitCnyAmountPattern, value, out var observedFen)
            && TryParseFullAmountFen(ExpectedCnyAmountPattern, expectedReceiptAmount, out var expectedFen)
            && observedFen == expectedFen;
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
}
