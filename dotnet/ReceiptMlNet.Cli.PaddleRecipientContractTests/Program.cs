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
                "\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551\uff09",
                RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)"),
                "mixed full-width payment card tail is repaired with full agreement");
            AssertTrue(
                ReceiptFieldNormalizer.IsMixedPaymentCardParenthesesCandidate(
                    "\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)"),
                "mixed payment candidate enters the auxiliary consensus path");
            AssertFalse(
                ReceiptFieldNormalizer.IsMixedPaymentCardParenthesesCandidate(
                    "\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361(2551)"),
                "valid ASCII payment candidate stays on the fast path");
            AssertFalse(
                ReceiptFieldNormalizer.IsMixedPaymentCardParenthesesCandidate(
                    "\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551\uff09"),
                "valid full-width payment candidate stays on the fast path");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361(2551)"), "valid ASCII payment parentheses");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551\uff09"), "valid full-width payment parentheses");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361(2551\uff09"), "reverse mixed payment parentheses");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff08255)", tail: "255"), "three-digit payment tail");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff08\uff12\uff15\uff15\uff11)"), "full-width payment digits");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff08255A)"), "letter in payment tail");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)tail"), "payment trailing text");
            AssertEqual(null, RepairPayment(" \u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)"), "payment leading whitespace");
            AssertEqual(null, RepairPayment("\u62db\u5546(\u94f6\u884c)\u50a8\u84c4\u5361\uff082551)", prefix: "\u62db\u5546(\u94f6\u884c)\u50a8\u84c4\u5361"), "payment prefix with parentheses");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)", prefix: "\u5de5\u5546\u94f6\u884c\u50a8\u84c4\u5361"), "payment prefix disagreement");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)", tail: "2552"), "payment tail disagreement");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)", structure: "unstructured"), "payment structure disagreement");
            AssertEqual(null, RepairPayment("\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361\uff082551)", style: "ascii"), "payment style disagreement");
            AssertAlternative(
                "\u53f8\u6e90(**\u6e90)",
                "pinyin_annotated_three_line",
                null,
                ParsePinyin(["Sh\u014du ku\u01cen f\u0101ng", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.80f, 0.91f, 0.80f], 0.90f),
                "strict pinyin annotation route");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], 0.899f),
                "pinyin detector below 0.90");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], float.NaN),
                "pinyin detector is non-finite");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], float.PositiveInfinity),
                "pinyin detector infinity is rejected");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.94f, float.NaN, 0.99f], 0.92f),
                "pinyin line confidence is non-finite");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.799f, 0.90f, 0.90f], 0.95f),
                "pinyin annotation line below 0.80");
            AssertAlternative(
                "\u53f8\u6e90(**\u6e90)",
                "pinyin_annotated_three_line",
                null,
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.70f, 0.90f], 0.95f),
                "pinyin merchant line at 0.70");
            AssertAlternative(
                "\u5c0f\u8363\uff08**\u9f99\uff09",
                "pinyin_annotated_three_line",
                null,
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.93031883f, 0.700153f, 0.9950261f],
                    0.92100316f),
                "exact calibrated pilot pinyin evidence");
            AssertAlternative(
                "\u5c0f\u8363\uff08**\u9f99\uff09",
                "pinyin_annotated_three_line_strong_anchors",
                null,
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.9424985f, 0.6764692f, 0.9966202f],
                    0.92100316f),
                "production rectified pinyin evidence with strong anchors");
            AssertAlternative(
                "\u5c0f\u8363\uff08**\u9f99\uff09",
                "pinyin_annotated_three_line_strong_anchors",
                null,
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.94f, 0.67f, 0.99f],
                    0.92f),
                "strong-anchor pinyin thresholds are inclusive");
            AssertAlternativeNull(
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.9424985f, 0.669f, 0.9966202f],
                    0.92100316f),
                "strong-anchor pinyin merchant line below 0.67");
            AssertAlternativeNull(
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.9424985f, 0.6764692f, 0.9966202f],
                    0.919f),
                "lower pinyin merchant confidence without strong detector anchor");
            AssertAlternativeNull(
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.939f, 0.6764692f, 0.9966202f],
                    0.92100316f),
                "lower pinyin merchant confidence without strong first-line anchor");
            AssertAlternativeNull(
                ParsePinyin(
                    ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                    [0.9424985f, 0.6764692f, 0.989f],
                    0.92100316f),
                "lower pinyin merchant confidence without strong label anchor");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.699f, 0.90f], 0.95f),
                "pinyin merchant line below 0.70");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u6536\u6b3e\u65b9", "\u53f8\u6e90(**\u6e90)"], [0.90f, 0.90f, 0.90f], 0.95f),
                "pinyin wrong line order");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan ren", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], 0.95f),
                "pinyin annotation typo");
            AssertAlternativeNull(
                ParsePinyin(["shou 1kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], 0.95f),
                "pinyin annotation contains non-letter noise");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "merchant-123", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], 0.95f),
                "pinyin value is not CJK");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u4eba"], [0.90f, 0.90f, 0.90f], 0.95f),
                "pinyin trailing label is not exact");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9", "\u5907\u6ce8"], [0.90f, 0.90f, 0.90f, 0.90f], 0.95f),
                "pinyin extra line");
            AssertAlternativeNull(
                ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)"], [0.90f, 0.90f], 0.95f),
                "pinyin missing label line");

            var strongPinyin = ParsePinyin(
                ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                [0.9424985f, 0.6764692f, 0.9966202f],
                0.92100316f);
            var ordinaryPinyin = ParsePinyin(
                ["shou kuan fang", "\u5c0f\u8363\uff08**\u9f99\uff09", "\u6536\u6b3e\u65b9"],
                [0.95f, 0.70f, 0.995f],
                0.95f);
            var differentStrongPinyin = ParsePinyin(
                ["shou kuan fang", "\u5c0f\u738b\uff08**\u6d77\uff09", "\u6536\u6b3e\u65b9"],
                [0.95f, 0.68f, 0.995f],
                0.95f);
            AssertTrue(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    strongPinyin, true, strongPinyin, true),
                "matching strong pinyin crops pass dual-crop gate");
            AssertTrue(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    strongPinyin, true, ordinaryPinyin, true),
                "matching strong and ordinary pinyin crops pass dual-crop gate");
            AssertFalse(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    strongPinyin, true, null, true),
                "single strong primary crop cannot pass dual-crop gate");
            AssertFalse(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    null, true, strongPinyin, true),
                "single strong retry crop cannot pass dual-crop gate");
            AssertFalse(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    strongPinyin, true, differentStrongPinyin, true),
                "different pinyin merchants cannot pass dual-crop gate");
            AssertFalse(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    strongPinyin, false, strongPinyin, true),
                "strong primary crop with failed geometry cannot pass dual-crop gate");
            AssertFalse(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    strongPinyin, true, strongPinyin, false),
                "strong retry crop with failed geometry cannot pass dual-crop gate");
            AssertFalse(
                PaddleRecipientValueParser.HasRequiredDualCropAgreement(
                    ordinaryPinyin, true, ordinaryPinyin, true),
                "ordinary pinyin crops do not use the low-confidence dual-crop gate");

            AssertAlternative(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                "unlabelled_cjk_amount_exact",
                0,
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00", "100.00", 0.75f),
                "CJK exact amount at 0.75");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00", "100.00", 0.749f),
                "CJK exact amount below 0.75");
            AssertAlternative(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                "unlabelled_cjk_amount_within_one_fen",
                1,
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.01", "100.00", 0.68f),
                "CJK one-fen drift at 0.68");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.01", "100.00", 0.679f),
                "CJK one-fen drift below 0.68");
            AssertAlternative(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                "unlabelled_cjk_amount_within_one_yuan",
                100,
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5101.00", "100.00", 0.90f),
                "CJK one-yuan drift at 0.90");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5101.00", "100.00", 0.899f),
                "CJK one-yuan drift below 0.90");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5101.01", "100.00", 0.99f),
                "CJK drift above one yuan");
            AssertAlternativeNull(
                ParsePair("\uffe5100.00", "\u6296\u97f3\u7535\u5546\u5546\u5bb6", "100.00", 0.99f),
                "reversed pair");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "100.00", "100.00", 0.99f),
                "amount without currency mark");
            AssertAlternativeNull(
                ParsePair("\u4ed8\u6b3e\u65b9\u5f0f", "\uffe5100.00", "100.00", 0.99f),
                "non-recipient row label");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00", "\u5907\u6ce8"],
                    [0.99f, 0.99f, 0.99f],
                    "100.00",
                    0.99f),
                "extra line");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    [0.799f, 0.99f],
                    "100.00",
                    0.99f),
                "merchant line below 0.80");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    [0.99f, 0.799f],
                    "100.00",
                    0.99f),
                "amount line below 0.80");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    [float.NaN, 0.99f],
                    "100.00",
                    0.99f),
                "non-finite pair line confidence");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    [0.99f],
                    "100.00",
                    0.99f),
                "pair confidence count mismatch");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    [0.99f, 0.99f],
                    "100.00",
                    float.NaN),
                "non-finite pair detector score");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
                    ["\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.00"],
                    [0.99f, 0.99f],
                    "100.00",
                    float.PositiveInfinity),
                "infinite pair detector score");
            AssertAlternativeNull(
                ParsePair("\u5546\u54c1\u540d\u79f0", "\uffe5100.00", "100.00", 0.99f),
                "product row");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5OO", "0.00", 0.99f),
                "OCR-confusable amount");
            AssertAlternative(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                "unlabelled_cjk_amount_exact",
                0,
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe51000.00", "1000.00", 0.75f),
                "full four-digit amount");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe51000.00", "2000.00", 0.99f),
                "four-digit amount mismatch above one yuan");
            AssertAlternativeNull(
                ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe51234.56", "9994.56", 0.99f),
                "shared-suffix amount mismatch");

            AssertAlternative(
                "123456",
                "unlabelled_numeric_amount_exact",
                0,
                ParsePair("123456", "\uffe5100.00", "100.00", 0.95f),
                "numeric merchant exact amount at 0.95");
            AssertAlternativeNull(ParsePair("123456", "\uffe5100.00", "100.00", 0.949f), "numeric merchant below 0.95");
            AssertAlternativeNull(ParsePair("1", "\uffe5100.00", "100.00", 0.99f), "numeric merchant too short");
            AssertAlternativeNull(ParsePair("123456789", "\uffe5100.00", "100.00", 0.99f), "numeric merchant too long");
            AssertAlternativeNull(ParsePair("12A34", "\uffe5100.00", "100.00", 0.99f), "numeric merchant not digits only");
            AssertAlternativeNull(ParsePair("123456", "\uffe5100.01", "100.00", 0.99f), "numeric merchant amount not exact");
            AssertAlternativeNull(ParsePair("100", "\uffe5100.00", "100.00", 0.99f), "numeric merchant equals amount integer part");
            AssertAlternativeNull(ParsePair("0100", "\uffe5100.00", "100.00", 0.99f), "zero-padded numeric merchant equals amount integer part");
            var exactCjkAlternative = ParsePair(
                "\u6296\u97f3\u7535\u5546\u5546\u5bb6",
                "\uffe5100.00",
                "100.00",
                0.84f);
            AssertTrue(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    exactCjkAlternative,
                    0.84f),
                "exact CJK overlap exception at 0.84");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    exactCjkAlternative,
                    0.839f),
                "exact CJK overlap exception below 0.84");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5100.01", "100.00", 0.99f),
                    0.99f),
                "one-fen route has no overlap exception");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParsePair("\u6296\u97f3\u7535\u5546\u5546\u5bb6", "\uffe5101.00", "100.00", 0.99f),
                    0.99f),
                "one-yuan route has no overlap exception");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParsePair("123456", "\uffe5100.00", "100.00", 0.99f),
                    0.99f),
                "numeric route has no overlap exception");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParsePinyin(["shou kuan fang", "\u53f8\u6e90(**\u6e90)", "\u6536\u6b3e\u65b9"], [0.90f, 0.90f, 0.90f], 0.99f),
                    0.99f),
                "pinyin route has no overlap exception");
            var amountBox = new[] { 200.0f, 300.0f, 550.0f, 420.0f };
            var recipientBox = new[] { 20.0f, 460.0f, 730.0f, 520.0f };
            var paymentBox = new[] { 180.0f, 550.0f, 720.0f, 610.0f };
            AssertTrue(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.68f, recipientBox, 0.80f, amountBox, 0.80f, paymentBox),
                "strict row geometry at calibrated floors");
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
            var paymentOverlapAt45Percent = new[] { 180.0f, 493.0f, 720.0f, 610.0f };
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.84f, recipientBox, 0.90f, amountBox, 0.90f, paymentOverlapAt45Percent),
                "45 percent payment overlap is rejected by default geometry");
            AssertTrue(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.84f,
                    recipientBox,
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentOverlapAt45Percent,
                    0.45f),
                "45 percent payment overlap at exact-route envelope");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.84f,
                    recipientBox,
                    0.90f,
                    amountBox,
                    0.90f,
                    [180.0f, 492.9f, 720.0f, 610.0f],
                    0.45f),
                "payment overlap above 45 percent");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.84f,
                    [20.0f, 390.0f, 730.0f, 450.0f],
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentBox,
                    0.45f),
                "exact-route exception does not relax amount overlap");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750,
                    1000,
                    0.84f,
                    recipientBox,
                    0.90f,
                    amountBox,
                    0.90f,
                    paymentOverlapAt45Percent,
                    0.451f),
                "payment overlap fraction above calibrated maximum");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, 0.679f, recipientBox, 0.90f, amountBox, 0.90f, paymentBox),
                "recipient detector below 0.68 floor");
            AssertFalse(
                PaddleRecipientValueParser.HasVerifiedUnlabelledMerchantRowGeometry(
                    750, 1000, float.NaN, recipientBox, 0.90f, amountBox, 0.90f, paymentBox),
                "non-finite recipient detector score");
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
            Console.WriteLine("PASS: PP-OCR recipient parsing is label-anchored or calibrated by pinyin/amount evidence and fail-closed.");
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

    private static PaddleRecipientAlternativeParseResult? ParsePinyin(
        IReadOnlyList<string> lines,
        IReadOnlyList<float> lineConfidences,
        float recipientDetectorScore)
    {
        return PaddleRecipientValueParser.ParsePinyinAnnotatedRecipient(
            lines,
            lineConfidences,
            recipientDetectorScore);
    }

    private static string? RepairPayment(
        string raw,
        string prefix = "\u62db\u5546\u94f6\u884c\u50a8\u84c4\u5361",
        string tail = "2551",
        string structure = "card_tail4",
        string style = "fullwidth")
    {
        return ReceiptFieldNormalizer.TryRepairMixedPaymentCardParentheses(
            raw,
            prefix,
            tail,
            structure,
            style);
    }

    private static PaddleRecipientAlternativeParseResult? ParsePair(
        string merchant,
        string amount,
        string expectedAmount,
        float score)
    {
        return PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(
            [merchant, amount],
            [0.99f, 0.99f],
            expectedAmount,
            score);
    }

    private static void AssertAlternative(
        string expectedValue,
        string expectedRoute,
        long? expectedAmountDeltaFen,
        PaddleRecipientAlternativeParseResult? actual,
        string label)
    {
        if (actual is null
            || !string.Equals(expectedValue, actual.Value, StringComparison.Ordinal)
            || !string.Equals(expectedRoute, actual.Route, StringComparison.Ordinal)
            || expectedAmountDeltaFen != actual.AmountDeltaFen)
        {
            throw new InvalidOperationException(
                $"{label}: expected '{expectedValue}' via '{expectedRoute}' with delta "
                + $"'{expectedAmountDeltaFen?.ToString() ?? "<null>"}', got "
                + $"'{actual?.Value ?? "<null>"}' via '{actual?.Route ?? "<null>"}' with delta "
                + $"'{actual?.AmountDeltaFen.ToString() ?? "<null>"}'");
        }
    }

    private static void AssertAlternativeNull(
        PaddleRecipientAlternativeParseResult? actual,
        string label)
    {
        if (actual is not null)
        {
            throw new InvalidOperationException(
                $"{label}: expected null, got '{actual.Value}' via '{actual.Route}'");
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
