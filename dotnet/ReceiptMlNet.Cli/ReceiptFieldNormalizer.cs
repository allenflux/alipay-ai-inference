using System;
using System.Globalization;
using System.Text.RegularExpressions;

/// <summary>
/// Conservative text cleanup and receipt-field normalisation used after the
/// PP-OCR ONNX adapter. This mirrors the business rules in
/// <c>transfer_receipt_ai.ocr</c>; it deliberately does not perform OCR.
/// </summary>
internal static class ReceiptFieldNormalizer
{
    private static readonly string[] StatusSuccessPhrases =
        ["\u8f6c\u8d26\u6210\u529f", "\u4ea4\u6613\u6210\u529f", "\u4ed8\u6b3e\u6210\u529f", "\u652f\u4ed8\u6210\u529f", "\u8f6c\u5e10\u6210\u529f"];
    private static readonly string[] StatusSuccessBlockingTokens =
        ["\u672a", "\u4e0d", "\u975e", "\u65e0", "\u5426", "\u6ca1", "\u6ca1\u6709", "\u672a\u80fd", "\u4e0d\u662f", "\u5e76\u672a", "\u5c1a\u672a", "\u4e0d\u80fd", "\u65e0\u6cd5", "\u6ca1\u80fd", "\u672a\u66fe", "\u4ece\u672a", "\u5e76\u975e", "\u5417", "\u4e48", "\u5f85\u786e\u8ba4", "\u5f85\u6838\u5b9e", "\u672a\u77e5", "\u4e0d\u786e\u5b9a", "\u7591\u4f3c"];
    private static readonly Regex WhitespacePattern = new(@"\s+", RegexOptions.CultureInvariant);
    private static readonly Regex AmountPattern = new(
        @"(?:[¥￥]\s*)?([0-9OoIl]{1,3}(?:,[0-9OoIl]{3})*(?:\.\d{1,2})?)",
        RegexOptions.CultureInvariant);
    private static readonly Regex TimePattern = new(
        @"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)",
        RegexOptions.CultureInvariant);
    private static readonly Regex MixedFullwidthPaymentCardTailPattern = new(
        @"^(?<prefix>[^()（）]+(?:银行卡|储蓄卡|信用卡))（(?<tail>[0-9]{4})\)$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly char[] FieldValueTrimCharacters = [' ', ':', '：', '-', '—'];

    /// <summary>Collapse all whitespace to one space and trim the result.</summary>
    public static string CleanText(string? value)
    {
        return string.IsNullOrEmpty(value) ? string.Empty : WhitespacePattern.Replace(value, " ").Trim();
    }

    /// <summary>
    /// Extract the value portion of an OCR'd recipient or payment-method row.
    /// If the value appears before its label, retain that preceding value.
    /// </summary>
    public static string ExtractFieldValue(string? rawText, string field)
    {
        var text = CleanText(rawText);
        var labels = field switch
        {
            "recipient" => new[] { "收款方", "收款人", "收款账户", "收款账号" },
            "payment_method" => new[] { "付款方式", "交易方式", "付款渠道", "支付方式" },
            _ => throw new ArgumentException($"Unknown field: {field}", nameof(field)),
        };

        foreach (var label in labels)
        {
            var position = text.IndexOf(label, StringComparison.Ordinal);
            if (position < 0)
            {
                continue;
            }

            var value = text[(position + label.Length)..].TrimStart(FieldValueTrimCharacters);
            if (value.Length > 0)
            {
                return value;
            }

            // OCR can emit the right-hand value before the row label.
            var valueBeforeLabel = text[..position].TrimEnd(FieldValueTrimCharacters);
            if (valueBeforeLabel.Length > 0)
            {
                return valueBeforeLabel;
            }
        }

        return text;
    }

    /// <summary>
    /// Extract a valid visible status-bar time without inventing a transaction
    /// timestamp.  Invalid OCR candidates are only reversed when the whole
    /// reversed token is a valid clock value.
    /// </summary>
    public static string? NormalizeTime(string? rawText)
    {
        var candidates = TimePattern.Matches(CleanText(rawText).Replace("：", ":", StringComparison.Ordinal));
        foreach (Match candidate in candidates)
        {
            if (IsValidStatusTime(candidate.Groups[1].Value))
            {
                return candidate.Groups[1].Value;
            }
        }

        foreach (Match candidate in candidates)
        {
            var reversed = Reverse(candidate.Groups[1].Value);
            if (IsValidStatusTime(reversed))
            {
                return reversed;
            }
        }

        return null;
    }

    /// <summary>Normalise a CNY amount while preserving its cleaned OCR text.</summary>
    public static NormalizedAmount? NormalizeAmount(string? rawText)
    {
        var raw = CleanText(rawText);
        var matches = AmountPattern.Matches(raw);
        if (matches.Count == 0)
        {
            return null;
        }

        Match? best = null;
        foreach (Match candidate in matches)
        {
            if (best is null || IsBetterAmountCandidate(candidate, best))
            {
                best = candidate;
            }
        }

        var numeric = best!.Groups[1].Value
            .Replace('O', '0')
            .Replace('o', '0')
            .Replace('I', '1')
            .Replace('l', '1')
            .Replace(",", string.Empty, StringComparison.Ordinal);
        if (!decimal.TryParse(
                numeric,
                NumberStyles.AllowDecimalPoint,
                CultureInfo.InvariantCulture,
                out var value))
        {
            return null;
        }

        value = decimal.Round(value, 2, MidpointRounding.AwayFromZero);
        if (value < 0m || value > decimal.Truncate(long.MaxValue / 100m))
        {
            return null;
        }

        var amountFen = decimal.ToInt64(value * 100m);
        return new NormalizedAmount(
            Raw: raw,
            Normalized: $"¥{value.ToString("0.00", CultureInfo.InvariantCulture)}",
            AmountFen: amountFen,
            Currency: "CNY");
    }

    /// <summary>Map recognisable Chinese transfer status text to a stable code.</summary>
    public static string NormalizeStatus(string? rawText)
    {
        var compact = WhitespacePattern.Replace(rawText ?? string.Empty, string.Empty);
        if (ContainsAny(compact, "失败", "未成功", "已撤销"))
        {
            return "failed";
        }
        if (ContainsAny(compact, "处理中", "待处理", "进行中"))
        {
            return "pending";
        }
        if (ContainsAny(compact, StatusSuccessPhrases))
        {
            // Treat any negation or uncertainty in the visible status string
            // as conflicting evidence.  A bounded prefix window misses both
            // suffixes and long-distance negation.
            if (ContainsAny(compact, StatusSuccessBlockingTokens))
            {
                return "unknown";
            }
            return "success";
        }
        return "unknown";
    }

    /// <summary>Map a cleaned payment-method row to its conservative category.</summary>
    public static NormalizedPaymentMethod NormalizePaymentMethod(string? rawText)
    {
        var raw = CleanText(rawText);
        var compact = WhitespacePattern.Replace(raw, string.Empty);
        var kind = compact.Contains("余额宝", StringComparison.Ordinal)
            ? "yuebao"
            : compact.Contains("余额", StringComparison.Ordinal)
                ? "balance"
                : compact.Contains("花呗", StringComparison.Ordinal)
                    ? "huabei"
                    : ContainsAny(compact, "银行卡", "储蓄卡", "信用卡")
                        ? "bank_card"
                        : "other";
        return new NormalizedPaymentMethod(raw, kind);
    }

    /// <summary>
    /// Repair only the observed mixed card-tail delimiter when all auxiliary
    /// heads agree with the raw CTC prefix and four digits. Valid ASCII or
    /// full-width pairs, the reverse mixed direction and character/digit
    /// disagreements deliberately remain unchanged by returning null.
    /// </summary>
    public static string? TryRepairMixedPaymentCardParentheses(
        string? rawCtc,
        string? prefixCtc,
        string? tailDigits,
        string? structureClass,
        string? parenthesesClass)
    {
        if (!IsMixedPaymentCardParenthesesCandidate(rawCtc)
            || string.IsNullOrEmpty(prefixCtc)
            || string.IsNullOrEmpty(tailDigits)
            || !string.Equals(prefixCtc, prefixCtc.Trim(), StringComparison.Ordinal)
            || !string.Equals(structureClass, "card_tail4", StringComparison.Ordinal)
            || !string.Equals(parenthesesClass, "fullwidth", StringComparison.Ordinal))
        {
            return null;
        }

        var raw = rawCtc!;
        var match = MixedFullwidthPaymentCardTailPattern.Match(raw);
        if (!match.Success
            || !string.Equals(match.Groups["prefix"].Value, prefixCtc, StringComparison.Ordinal)
            || !string.Equals(match.Groups["tail"].Value, tailDigits, StringComparison.Ordinal))
        {
            return null;
        }
        return raw[..^1] + "）";
    }

    public static bool IsMixedPaymentCardParenthesesCandidate(string? rawCtc)
    {
        return !string.IsNullOrEmpty(rawCtc)
            && string.Equals(rawCtc, rawCtc.Trim(), StringComparison.Ordinal)
            && MixedFullwidthPaymentCardTailPattern.IsMatch(rawCtc);
    }

    private static bool IsBetterAmountCandidate(Match candidate, Match current)
    {
        var candidateCurrency = HasCurrencyMark(candidate);
        var currentCurrency = HasCurrencyMark(current);
        if (candidateCurrency != currentCurrency)
        {
            return candidateCurrency;
        }

        var candidateDecimal = candidate.Groups[1].Value.IndexOf('.') >= 0;
        var currentDecimal = current.Groups[1].Value.IndexOf('.') >= 0;
        if (candidateDecimal != currentDecimal)
        {
            return candidateDecimal;
        }

        // Preserve Python's max() tie behaviour: the earlier match wins.
        return candidate.Groups[1].Value.Length > current.Groups[1].Value.Length;
    }

    private static bool HasCurrencyMark(Match match)
    {
        var token = match.Value.TrimStart();
        return token.StartsWith("¥", StringComparison.Ordinal) || token.StartsWith("￥", StringComparison.Ordinal);
    }

    private static bool IsValidStatusTime(string candidate)
    {
        var values = candidate.Split(':');
        if (values.Length is not (2 or 3)
            || !int.TryParse(values[0], NumberStyles.None, CultureInfo.InvariantCulture, out var hour)
            || !int.TryParse(values[1], NumberStyles.None, CultureInfo.InvariantCulture, out var minute))
        {
            return false;
        }

        var seconds = 0;
        if (values.Length == 3
            && !int.TryParse(values[2], NumberStyles.None, CultureInfo.InvariantCulture, out seconds))
        {
            return false;
        }
        return hour is >= 0 and <= 23
            && minute is >= 0 and <= 59
            && seconds is >= 0 and <= 59;
    }

    private static bool ContainsAny(string value, params string[] tokens)
    {
        foreach (var token in tokens)
        {
            if (value.Contains(token, StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    private static string Reverse(string value)
    {
        var characters = value.ToCharArray();
        Array.Reverse(characters);
        return new string(characters);
    }
}

/// <summary>Structured representation of a recognised CNY amount.</summary>
internal sealed record NormalizedAmount(string Raw, string Normalized, long AmountFen, string Currency);

/// <summary>Structured representation of a recognised payment method.</summary>
internal sealed record NormalizedPaymentMethod(string Raw, string Normalized);
