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

            AssertAlternative(
                "\u53f8\u6e90(**\u6e90)",
                "unlabelled_masked_cjk_right_full_agreement",
                null,
                ParseMaskedAgreement(
                    "\u53f8\u6e90(**\u6e90)",
                    "\u53f8\u6e90(**\u6e90)",
                    fullConfidence: 0.91f,
                    rightConfidence: 0.83f),
                "masked ASCII-parenthesis full/right agreement",
                0.83f);
            AssertAlternative(
                "\u5c0f\u8363\uff08**\u9f99\uff09",
                "unlabelled_masked_cjk_right_full_agreement",
                null,
                ParseMaskedAgreement(
                    "\u5c0f\u8363\uff08**\u9f99\uff09",
                    "\u5c0f\u8363\uff08**\u9f99\uff09",
                    fullConfidence: 0.80f,
                    rightConfidence: 0.80f,
                    fullWidth: 675,
                    rightWidth: 438,
                    detectorScore: 0.95f),
                "masked route inclusive confidence/crop/detector thresholds",
                0.80f);
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6d77)"),
                "masked full/right string difference");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", fullConfidence: 0.799f),
                "masked full confidence below 0.80");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", rightConfidence: 0.799f),
                "masked right confidence below 0.80");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", fullConfidence: float.NaN),
                "masked non-finite full confidence");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", rightConfidence: float.PositiveInfinity),
                "masked infinite right confidence");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", detectorScore: 0.949f),
                "masked detector below 0.95");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", detectorScore: float.NaN),
                "masked non-finite detector");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", detectorScore: float.PositiveInfinity),
                "masked infinite detector");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", fullWidth: 674),
                "masked full crop below 90 percent source width");
            AssertAlternativeNull(
                ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)", fullWidth: 675, rightWidth: 439),
                "masked right crop above 65 percent of full crop");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledMaskedCjkRightFullAgreement(
                    ["\u53f8\u6e90(**\u6e90)", "\u5907\u6ce8"],
                    [0.90f, 0.90f],
                    720,
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.90f],
                    396,
                    750,
                    0.95f),
                "masked full crop has an extra non-empty line");
            foreach (var invalidMasked in new[]
            {
                "\u53f8\u6e90()",
                "\u53f8\u6e90\uff08\uff09",
                "\u53f8\u6e90(\u6e90\u6d77)",
                "\u53f8\u6e90(**\u6e90\uff09",
                "\u53f8\u6e90\uff08**\u6e90)",
                "\u53f8(*\u6e90)",
                "\u53f8\u6e90ABCDEFGHIJK(**)",
                "Merchant(**)",
                "\u53f8\u6e90(*1)",
                "\u53f8\u6e90(*******)",
                "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u7532\u4e59\u4e19(**)",
                "\u5904\u7406(**)",
                "\u8be6\u60c5(**)",
                "\u8d26\u53f7(**)",
                "\u8d26\u6237(**)",
            })
            {
                AssertAlternativeNull(
                    ParseMaskedAgreement(invalidMasked, invalidMasked),
                    $"invalid masked merchant '{invalidMasked}'");
            }

            AssertAlternative(
                "\u8d8a\u91ce\u5154\uff08\uff09",
                "truncated_recipient_label_empty_mask_three_crop_agreement",
                null,
                ParseEmptyMaskAgreement(
                    "\u6536\u6b3e",
                    "\u8d8a\u91ce\u5154\uff08\uff09",
                    "\u8d8a\u91ce\u5154\uff08\uff09",
                    "\u8d8a\u91ce\u5154\uff08\uff09",
                    labelConfidence: 0.50f,
                    standardCandidateConfidence: 0.82f,
                    fullConfidence: 0.76f,
                    rightConfidence: 0.75f),
                "empty-mask three-crop agreement at inclusive thresholds",
                0.75f);
            AssertAlternative(
                "\u8d8a\u91ce\u5154()",
                "truncated_recipient_label_empty_mask_three_crop_agreement",
                null,
                ParseEmptyMaskAgreement(
                    "\u6536\u6b3e",
                    "\u8d8a\u91ce\u5154()",
                    "\u8d8a\u91ce\u5154()",
                    "\u8d8a\u91ce\u5154()"),
                "empty-mask ASCII parentheses",
                0.90f);
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e\u65b9", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09"),
                "empty-mask standard label must be exactly truncated shoukuan");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u72d0\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09"),
                "empty-mask full crop differs");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u72d0\uff08\uff09"),
                "empty-mask right crop differs");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", labelConfidence: 0.499f),
                "empty-mask label confidence below 0.50");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", standardCandidateConfidence: 0.799f),
                "empty-mask standard candidate below 0.80");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", fullConfidence: 0.749f),
                "empty-mask full confidence below 0.75");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", rightConfidence: 0.749f),
                "empty-mask right confidence below 0.75");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", labelConfidence: float.NaN),
                "empty-mask non-finite label confidence");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", detectorScore: float.PositiveInfinity),
                "empty-mask infinite detector");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", fullWidth: 674),
                "empty-mask full crop below 90 percent source width");
            AssertAlternativeNull(
                ParseEmptyMaskAgreement("\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", "\u8d8a\u91ce\u5154\uff08\uff09", fullWidth: 675, rightWidth: 439),
                "empty-mask right crop above 65 percent of full crop");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseTruncatedRecipientLabelEmptyMaskThreeCropAgreement(
                    ["\u6536\u6b3e", "\u8d8a\u91ce\u5154\uff08\uff09", "\u5907\u6ce8"],
                    [0.90f, 0.90f, 0.90f],
                    ["\u8d8a\u91ce\u5154\uff08\uff09"],
                    [0.90f],
                    720,
                    ["\u8d8a\u91ce\u5154\uff08\uff09"],
                    [0.90f],
                    396,
                    750,
                    0.95f),
                "empty-mask standard crop has extra line");
            foreach (var invalidEmptyMask in new[]
            {
                "\u8d8a\u91ce\u5154(**)",
                "\u8d8a\u91ce\u5154(\uff09",
                "\u8d8a\u91ce\u5154\uff08)",
                "\u8d8a()",
                "\u8d8a\u91ce\u5154\u5546\u6237\u540d\u79f0\u8d85\u957f()",
                "A\u8d8a\u91ce\u5154()",
                "\u5904\u7406()",
                "\u8be6\u60c5()",
                "\u8d26\u53f7()",
                "\u8d26\u6237()",
            })
            {
                AssertAlternativeNull(
                    ParseEmptyMaskAgreement(
                        "\u6536\u6b3e",
                        invalidEmptyMask,
                        invalidEmptyMask,
                        invalidEmptyMask),
                    $"invalid empty-mask merchant '{invalidEmptyMask}'");
            }

            AssertAlternative(
                "\u5f20\u5976\u6c38",
                "unlabelled_cjk_discount_arithmetic_exact",
                8,
                ParseDiscountArithmetic(
                    "\u5f20\u5976\u6c38",
                    "\u00a5 500.00",
                    "\u767e\u6b21\u7acb\u51cf",
                    "08",
                    "499.92",
                    [0.99f, 0.90f, 0.99f, 0.99f],
                    0.95f),
                "exact four-line discount arithmetic evidence",
                0.90f);
            AssertAlternative(
                "\u5f20\u00b7\u5976\u6c38",
                "unlabelled_cjk_discount_arithmetic_exact",
                1,
                ParseDiscountArithmetic("\u5f20\u00b7\u5976\u6c38", "\uffe5500.00", "\u767e\u6b21\u7acb\u51cf", "01", "499.99"),
                "discount route permits internal middle dot",
                0.99f);
            AssertAlternative(
                "\u5f20\u5976\u6c38",
                "unlabelled_cjk_discount_arithmetic_exact",
                99,
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "99", "499.01"),
                "discount 99-fen upper boundary",
                0.99f);
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.91"),
                "discount arithmetic mismatch");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "500.08"),
                "gross must exceed expected amount");
            foreach (var invalidGross in new[]
            {
                "500.00",
                "\u00a5500.0",
                "\u00a5500.000",
                "\u00a51,500.00",
                "\u00a5\uff15\uff10\uff10.\uff10\uff10",
                "\u00a5OOO.00",
            })
            {
                AssertAlternativeNull(
                    ParseDiscountArithmetic("\u5f20\u5976\u6c38", invalidGross, "\u767e\u6b21\u7acb\u51cf", "08", "499.92"),
                    $"invalid strict discount gross '{invalidGross}'");
            }
            foreach (var invalidDiscount in new[] { "00", "1", "100", "\uff10\uff18", "0A" })
            {
                AssertAlternativeNull(
                    ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", invalidDiscount, "499.92"),
                    $"invalid exact two-digit discount '{invalidDiscount}'");
            }
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u6bcf\u6b21\u7acb\u51cf", "08", "499.92"),
                "discount marker must be exact");
            foreach (var invalidMerchant in new[]
            {
                "A\u5f20\u5976\u6c38",
                "\u00b7\u5f20\u5976\u6c38",
                "\u5f20\u5976\u6c38\u00b7",
                "\u5f20\u00b7\u00b7\u5976\u6c38",
                "\u5f20\u5976\u6c38500",
                "\u5904\u7406\u8be6\u60c5",
                "\u8d26\u53f7\u4fe1\u606f",
                "\u8d26\u6237\u4fe1\u606f",
            })
            {
                AssertAlternativeNull(
                    ParseDiscountArithmetic(invalidMerchant, "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92"),
                    $"invalid discount merchant '{invalidMerchant}'");
            }
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", [0.989f, 0.99f, 0.99f, 0.99f]),
                "discount merchant confidence below 0.99");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", [0.99f, 0.899f, 0.99f, 0.99f]),
                "discount gross confidence below 0.90");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", [0.99f, 0.99f, 0.989f, 0.99f]),
                "discount marker confidence below 0.99");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", [0.99f, 0.99f, 0.99f, 0.989f]),
                "discount fen confidence below 0.99");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", [0.99f, float.NaN, 0.99f, 0.99f]),
                "discount non-finite line confidence");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", detectorScore: 0.949f),
                "discount detector below 0.95");
            AssertAlternativeNull(
                ParseDiscountArithmetic("\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "499.92", detectorScore: float.NaN),
                "discount non-finite detector");
            AssertAlternativeNull(
                PaddleRecipientValueParser.ParseUnlabelledCjkDiscountArithmeticExact(
                    ["\u5f20\u5976\u6c38", "\u00a5500.00", "\u767e\u6b21\u7acb\u51cf", "08", "\u5907\u6ce8"],
                    [0.99f, 0.99f, 0.99f, 0.99f, 0.99f],
                    "499.92",
                    0.95f),
                "discount route extra line");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParseMaskedAgreement("\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)"),
                    0.99f),
                "masked full/right route cannot use 45-percent overlap exception");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParseEmptyMaskAgreement(
                        "\u6536\u6b3e",
                        "\u8d8a\u91ce\u5154\uff08\uff09",
                        "\u8d8a\u91ce\u5154\uff08\uff09",
                        "\u8d8a\u91ce\u5154\uff08\uff09"),
                    0.99f),
                "empty-mask route cannot use 45-percent overlap exception");
            AssertFalse(
                PaddleRecipientValueParser.AllowsExactCjkPaymentOverlapException(
                    ParseDiscountArithmetic(
                        "\u5f20\u5976\u6c38",
                        "\u00a5 500.00",
                        "\u767e\u6b21\u7acb\u51cf",
                        "08",
                        "499.92"),
                    0.99f),
                "discount arithmetic route cannot use 45-percent overlap exception");

            AssertAlternative(
                "\u53f8\u6e90(**\u6e90)",
                "independent_crop_exact_consensus",
                null,
                ParseConsensus(
                    ["  \u53f8\u6e90(**\u6e90)  ", "\u4ed8\u6b3e\u65b9\u5f0f"],
                    [0.80f, 0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.91f],
                    null,
                    null),
                "independent-crop exact consensus at inclusive line floor",
                0.80f);
            AssertAlternative(
                "Merchant-123",
                "independent_crop_exact_consensus",
                null,
                ParseConsensus(
                    ["Merchant-123"],
                    [0.93f],
                    ["different"],
                    [0.99f],
                    ["Merchant-123"],
                    [0.88f]),
                "first and right-value crops can form exact consensus",
                0.88f);
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.799f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null),
                "consensus participating line below 0.80");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null,
                    detectorScore: 0.679f),
                "consensus recipient detector below 0.68");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null,
                    detectorScore: float.NaN),
                "consensus non-finite recipient detector");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null,
                    detectorScore: 1.001f),
                "consensus recipient detector above one");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [1.001f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null),
                "consensus participating line above one");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null,
                    ordinaryGeometryVerified: false),
                "consensus ordinary 25-percent geometry must be verified");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    null,
                    null,
                    alternativeEnvelopeVerified: false),
                "consensus alternative envelope must be verified");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.99f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.99f],
                    null,
                    null),
                "multiple exact consensus candidates are ambiguous");
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.95f]),
                "ordinary exact consensus remains ambiguous before 3-of-3 fallback");
            AssertAlternative(
                "\u53f8\u6e90(**\u6e90)",
                "independent_crop_dominant_three_crop_consensus",
                null,
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["  \u53f8\u6e90(**\u6e90)  "],
                    [0.80f]),
                "unique eligible candidate across all three existing crops",
                0.80f);
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.99f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.97f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.95f]),
                "sole 3-of-3 candidate belongs to earlier exact consensus route");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.95f, 0.94f]),
                "two 3-of-3 candidates remain ambiguous");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.799f]),
                "dominant candidate must pass line floor in the third crop");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.95f],
                    detectorScore: 0.679f),
                "dominant consensus cannot bypass recipient detector floor");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.95f],
                    ordinaryGeometryVerified: false),
                "dominant consensus cannot bypass ordinary 25-percent geometry");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    [0.95f],
                    alternativeEnvelopeVerified: false),
                "dominant consensus cannot bypass alternative envelope");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["Payment Method", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["Payment Method", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["Payment Method"],
                    [0.95f]),
                "dominant consensus preserves the exact UI and value filters");
            AssertAlternativeNull(
                ParseDominantConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.99f, 0.98f],
                    ["\u53f8\u6e90(**\u6e90)", "\u5c0f\u738b(**\u6d77)"],
                    [0.97f, 0.96f],
                    ["\u53f8\u6e90(**\u6e90)"],
                    []),
                "dominant consensus rejects line-confidence count mismatch");
            foreach (var forbiddenConsensusLine in new[]
            {
                "\u00a5200.00",
                "$200.00",
                "200.00\u5143",
                "CNY 200.00",
                "05:49",
                "\u90ae\u50a8\u94f6\u884c\u50a8\u84c4\u5361(8885)",
                "\u8f6c\u8d26\u6210\u529f",
                "\u6d3b\u52a8\u5956\u52b1",
                "shoukuanfang",
                "shou kuan ting",
                "shoukudnfang",
                "Payment Method",
                "Transfer Success",
                "Recipient",
                "Payee",
                "Amount",
                "Time",
                "Status",
                "Transfer Failed",
                "Processing",
                "Bank Card",
                "\u53f8\u6e90\ud83d\ude42",
            })
            {
                AssertAlternativeNull(
                    ParseConsensus(
                        [forbiddenConsensusLine],
                        [0.99f],
                        [forbiddenConsensusLine],
                        [0.99f],
                        null,
                        null),
                    $"forbidden consensus line '{forbiddenConsensusLine}'");
            }
            AssertAlternative(
                "jia",
                "independent_crop_exact_consensus",
                null,
                ParseConsensus(
                    ["jia"],
                    [0.91f],
                    ["jia"],
                    [0.90f],
                    null,
                    null),
                "short opaque ASCII payee is not rejected as a pinyin label",
                0.90f);
            AssertAlternative(
                "Success Store",
                "independent_crop_exact_consensus",
                null,
                ParseConsensus(
                    ["Success Store"],
                    [0.91f],
                    ["Success Store"],
                    [0.90f],
                    null,
                    null),
                "ASCII UI keys are exact whole-line matches, not substrings",
                0.90f);
            AssertAlternativeNull(
                ParseConsensus(
                    ["\u53f8\u6e90(**\u6e90)", "\u53f8\u6e90(**\u6e90)"],
                    [0.99f, 0.99f],
                    null,
                    null,
                    null,
                    null),
                "duplicate lines within one crop are not independent evidence");

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

    private static PaddleRecipientAlternativeParseResult? ParseMaskedAgreement(
        string full,
        string right,
        float fullConfidence = 0.90f,
        float rightConfidence = 0.90f,
        int fullWidth = 720,
        int rightWidth = 396,
        int sourceWidth = 750,
        float detectorScore = 0.95f)
    {
        return PaddleRecipientValueParser.ParseUnlabelledMaskedCjkRightFullAgreement(
            [full],
            [fullConfidence],
            fullWidth,
            [right],
            [rightConfidence],
            rightWidth,
            sourceWidth,
            detectorScore);
    }

    private static PaddleRecipientAlternativeParseResult? ParseEmptyMaskAgreement(
        string label,
        string standardCandidate,
        string fullCandidate,
        string rightCandidate,
        float labelConfidence = 0.90f,
        float standardCandidateConfidence = 0.90f,
        float fullConfidence = 0.90f,
        float rightConfidence = 0.90f,
        int fullWidth = 720,
        int rightWidth = 396,
        int sourceWidth = 750,
        float detectorScore = 0.95f)
    {
        return PaddleRecipientValueParser.ParseTruncatedRecipientLabelEmptyMaskThreeCropAgreement(
            [label, standardCandidate],
            [labelConfidence, standardCandidateConfidence],
            [fullCandidate],
            [fullConfidence],
            fullWidth,
            [rightCandidate],
            [rightConfidence],
            rightWidth,
            sourceWidth,
            detectorScore);
    }

    private static PaddleRecipientAlternativeParseResult? ParseDiscountArithmetic(
        string merchant,
        string gross,
        string marker,
        string discountFen,
        string expectedAmount,
        IReadOnlyList<float>? lineConfidences = null,
        float detectorScore = 0.95f)
    {
        return PaddleRecipientValueParser.ParseUnlabelledCjkDiscountArithmeticExact(
            [merchant, gross, marker, discountFen],
            lineConfidences ?? [0.99f, 0.99f, 0.99f, 0.99f],
            expectedAmount,
            detectorScore);
    }

    private static PaddleRecipientAlternativeParseResult? ParseConsensus(
        IReadOnlyList<string>? firstLines,
        IReadOnlyList<float>? firstConfidences,
        IReadOnlyList<string>? retryLines,
        IReadOnlyList<float>? retryConfidences,
        IReadOnlyList<string>? rightValueLines,
        IReadOnlyList<float>? rightValueConfidences,
        float detectorScore = 0.95f,
        bool ordinaryGeometryVerified = true,
        bool alternativeEnvelopeVerified = true)
    {
        return PaddleRecipientValueParser.ParseIndependentCropExactConsensus(
            firstLines,
            firstConfidences,
            retryLines,
            retryConfidences,
            rightValueLines,
            rightValueConfidences,
            detectorScore,
            ordinaryGeometryVerified,
            alternativeEnvelopeVerified);
    }

    private static PaddleRecipientAlternativeParseResult? ParseDominantConsensus(
        IReadOnlyList<string>? firstLines,
        IReadOnlyList<float>? firstConfidences,
        IReadOnlyList<string>? retryLines,
        IReadOnlyList<float>? retryConfidences,
        IReadOnlyList<string>? rightValueLines,
        IReadOnlyList<float>? rightValueConfidences,
        float detectorScore = 0.95f,
        bool ordinaryGeometryVerified = true,
        bool alternativeEnvelopeVerified = true)
    {
        return PaddleRecipientValueParser.ParseIndependentCropDominantThreeCropConsensus(
            firstLines,
            firstConfidences,
            retryLines,
            retryConfidences,
            rightValueLines,
            rightValueConfidences,
            detectorScore,
            ordinaryGeometryVerified,
            alternativeEnvelopeVerified);
    }

    private static void AssertAlternative(
        string expectedValue,
        string expectedRoute,
        long? expectedAmountDeltaFen,
        PaddleRecipientAlternativeParseResult? actual,
        string label,
        float? expectedCandidateConfidence = null)
    {
        if (actual is null
            || !string.Equals(expectedValue, actual.Value, StringComparison.Ordinal)
            || !string.Equals(expectedRoute, actual.Route, StringComparison.Ordinal)
            || expectedAmountDeltaFen != actual.AmountDeltaFen
            || (expectedCandidateConfidence is { } expectedConfidence
                && (actual.CandidateConfidence is not { } actualConfidence
                    || Math.Abs(actualConfidence - expectedConfidence) > 0.000001f)))
        {
            throw new InvalidOperationException(
                $"{label}: expected '{expectedValue}' via '{expectedRoute}' with delta "
                + $"'{expectedAmountDeltaFen?.ToString() ?? "<null>"}' and confidence "
                + $"'{expectedCandidateConfidence?.ToString() ?? "<unasserted>"}', got "
                + $"'{actual?.Value ?? "<null>"}' via '{actual?.Route ?? "<null>"}' with delta "
                + $"'{actual?.AmountDeltaFen.ToString() ?? "<null>"}' and confidence "
                + $"'{actual?.CandidateConfidence.ToString() ?? "<null>"}'");
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
