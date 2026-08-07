/// <summary>
/// Strict extraction contract shared by the PP-OCR recipient route and its
/// package-free .NET contract test.
/// </summary>
internal static class PaddleRecipientValueParser
{
    private static readonly string[] RecipientLabels =
        ["\u6536\u6b3e\u65b9", "\u6536\u6b3e\u4eba", "\u6536\u6b3e\u8d26\u6237", "\u6536\u6b3e\u8d26\u53f7"];
    private static readonly char[] RowSeparators = [' ', ':', '\uff1a', '-', '\u2014'];

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
}
