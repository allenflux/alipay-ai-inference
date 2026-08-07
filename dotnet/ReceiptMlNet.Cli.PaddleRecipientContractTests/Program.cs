internal static class Program
{
    private static int Main()
    {
        try
        {
            AssertEqual("\u53f8\u6e90(**\u6e90)", PaddleRecipientValueParser.Parse("\u6536\u6b3e\u65b9 \u53f8\u6e90(**\u6e90)"), "ordinary row");
            AssertEqual("\u5546\u6237\u7532", PaddleRecipientValueParser.Parse("  \u6536\u6b3e\u4eba\uff1a\u5546\u6237\u7532  "), "full-width colon");
            AssertEqual("6222****0000", PaddleRecipientValueParser.Parse("\u6536\u6b3e\u8d26\u6237--6222****0000"), "account row");
            AssertEqual(null, PaddleRecipientValueParser.Parse("\u5546\u6237\u7532 \u6536\u6b3e\u65b9"), "value before label");
            AssertEqual(null, PaddleRecipientValueParser.Parse("\u5907\u6ce8 \u6536\u6b3e\u65b9 \u5546\u6237\u7532"), "middle label");
            AssertEqual(null, PaddleRecipientValueParser.Parse("\u6536\u6b3e\u65b9\uff1a"), "empty value");
            AssertEqual(null, PaddleRecipientValueParser.Parse(null), "null OCR");
            Console.WriteLine("PASS: PP-OCR recipient parsing is strict, left-anchored and fail-closed.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void AssertEqual(string? expected, string? actual, string label)
    {
        if (!string.Equals(expected, actual, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{label}: expected '{expected ?? "<null>"}', got '{actual ?? "<null>"}'");
        }
    }
}
