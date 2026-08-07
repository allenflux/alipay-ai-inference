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
            AssertEqual(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    "100.00"),
                "label-free merchant plus explicit CNY amount");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\uffe5100.00", "\u6296\u97f3\u7535\u5546\u5546\u5bb6"],
                    "100.00"),
                "reversed pair");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "100.00"],
                    "100.00"),
                "amount without currency mark");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u4ed8\u6b3e\u65b9\u5f0f", "\uffe5100.00"],
                    "100.00"),
                "non-recipient row label");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00", "\u5907\u6ce8"],
                    "100.00"),
                "extra line");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u5546\u54c1\u540d\u79f0", "\uffe5100.00"],
                    "100.00"),
                "product row");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    "99.00"),
                "amount mismatch");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5OO"],
                    "0.00"),
                "OCR-confusable amount");
            AssertEqual(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe51000.00"],
                    "1000.00"),
                "full four-digit amount");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe51000.00"],
                    "2000.00"),
                "four-digit amount mismatch");
            AssertEqual(
                null,
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe51234.56"],
                    "9994.56"),
                "shared-suffix amount mismatch");
            var amountBox = new[] { 200.0f, 300.0f, 550.0f, 420.0f };
            var recipientBox = new[] { 20.0f, 460.0f, 730.0f, 520.0f };
            var paymentBox = new[] { 180.0f, 550.0f, 720.0f, 610.0f };
            AssertTrue(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.95f, recipientBox, 0.90f, amountBox, 0.90f, paymentBox),
                "strict row geometry");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.95f,
                    [20.0f, 380.0f, 730.0f, 500.0f],
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentBox),
                "amount overlap");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.95f,
                    [20.0f, 500.0f, 730.0f, 590.0f],
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentBox),
                "payment overlap");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.89f, recipientBox, 0.90f, amountBox, 0.90f, paymentBox),
                "low recipient detector confidence");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.95f, recipientBox, 0.79f, amountBox, 0.90f, paymentBox),
                "low amount detector confidence");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.95f, recipientBox, 0.90f, amountBox, 0.79f, paymentBox),
                "low payment detector confidence");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.95f,
                    [200.0f, 460.0f, 550.0f, 520.0f],
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentBox),
                "recipient row is not full width");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.95f,
                    [20.0f, 430.0f, 730.0f, 590.0f],
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentBox),
                "recipient row is implausibly tall");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.95f,
                    [20.0f, float.NaN, 730.0f, 520.0f],
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentBox),
                "non-finite geometry");
            Console.WriteLine("PASS: PP-OCR recipient parsing is strict, label-anchored or amount-paired, and fail-closed.");
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

    private static void AssertTrue(bool actual, string label)
    {
        if (!actual)
        {
            throw new InvalidOperationException($"{label}: expected true, got false");
        }
    }

    private static void AssertFalse(bool actual, string label)
    {
        if (actual)
        {
            throw new InvalidOperationException($"{label}: expected false, got true");
        }
    }
}
