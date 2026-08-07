using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

/// <summary>
/// Immutable, verified delivery description for a v12/v13 unified receipt
/// reader.  The reader is deliberately loaded only from the adjacent ONNX,
/// labels, and contract sidecars: a hand-edited model or mismatched decoder
/// vocabulary must fail before a session is created.
/// </summary>
internal sealed class UnifiedOcrBundle
{
    public const string KindV12 = "receipt_unified_field_reader_v12";
    public const string KindV13 = "receipt_unified_field_reader_v13";
    public const string StatusTextOutputName = "status_text_logits";
    public const string StatusTextRuntimePolicy = "decode_and_normalize_review_only";
    public const string StatusTextTarget = "visible_transfer_status_cjk_text";
    public const string StatusTextCharsetSource = "train_only_visible_transfer_status_cjk_text";
    public const string StatusTextNormalizer = "normalize_status";

    public static readonly string[] SlotOrder =
    [
        "amount",
        "time",
        "transfer_status",
        "payment_method_field",
        "recipient_field",
    ];

    private static readonly string[] V12OutputNames =
    [
        "amount_logits",
        "time_logits",
        "payment_logits",
        "status_logits",
        "amount_currency_style_logits",
        "amount_grouped_thousands_logits",
        "amount_sign_position_logits",
        "time_format_logits",
        "time_digit_logits",
        "payment_prefix_logits",
        "payment_bank_prefix_logits",
        "payment_tail_digit_logits",
        "payment_structure_logits",
        "payment_parentheses_logits",
        "recipient_logits",
    ];

    private static readonly string[] V13OutputNames =
    [
        .. V12OutputNames,
        StatusTextOutputName,
    ];

    private static readonly string[] ExpectedAmountCharacters = "0123456789.-".Select(value => value.ToString()).ToArray();
    private static readonly string[] ExpectedTimeCharacters = "0123456789:- ".Select(value => value.ToString()).ToArray();
    private static readonly string[] ExpectedStatusClasses = ["success", "pending", "failed"];
    private static readonly string[] ExpectedAmountCurrencyStyles = ["none", "yen", "yen_space", "fullwidth_yen", "fullwidth_yen_space"];
    private static readonly string[] ExpectedAmountGroupingStyles = ["ungrouped", "grouped_thousands"];
    private static readonly string[] ExpectedAmountSignPositions = ["none", "before_currency_or_number", "after_currency"];
    private static readonly string[] ExpectedTimeFormats =
    [
        "clock_h_mm",
        "clock_hh_mm",
        "clock_h_mm_ss",
        "clock_hh_mm_ss",
        "date_ymd_hh_mm",
        "date_ymd_hh_mm_ss",
    ];
    private static readonly string[] ExpectedPaymentStructureClasses = ["unstructured", "card_tail4"];
    private static readonly string[] ExpectedPaymentParenthesisClasses = ["ascii", "fullwidth"];
    private const string TextReviewPolicy = "review_only_pending_independent_human_truth_calibration";
    private const string ReviewValue = "review";
    private const string RecipientTarget = "anchored_recipient_value_with_dedicated_high_resolution_value_view";
    private const string RecipientPreprocess = "left_trim_then_centered_aspect_resize_high_resolution";
    private const string RecipientCharsetSource = "train_only_anchored_recipient_value";
    private const string WhitePlaceholderPolicy = "white_placeholder_not_decoded; emit review instead";

    private UnifiedOcrBundle(
        string modelPath,
        int architectureVersion,
        string labelsPath,
        string contractPath,
        string modelSha256,
        string labelsSha256,
        string contractSha256,
        int fieldHeight,
        int fieldWidth,
        int recipientHeight,
        int recipientWidth,
        double recipientLeftTrim,
        float amountFormatMinimumConfidence,
        IReadOnlyList<string> amountCharacters,
        IReadOnlyList<string> timeCharacters,
        IReadOnlyList<string> paymentCharacters,
        IReadOnlyList<string> recipientCharacters,
        IReadOnlyList<string> paymentBankPrefixClasses,
        IReadOnlyList<string> statusClasses,
        IReadOnlyList<string>? statusTextCharacters,
        IReadOnlyDictionary<string, int[]> outputShapes,
        string statusRuntimePolicy,
        string? statusReviewValue,
        string textRuntimePolicy,
        string textReviewValue)
    {
        ModelPath = modelPath;
        ArchitectureVersion = architectureVersion;
        LabelsPath = labelsPath;
        ContractPath = contractPath;
        ModelSha256 = modelSha256;
        LabelsSha256 = labelsSha256;
        ContractSha256 = contractSha256;
        FieldHeight = fieldHeight;
        FieldWidth = fieldWidth;
        RecipientHeight = recipientHeight;
        RecipientWidth = recipientWidth;
        RecipientLeftTrim = recipientLeftTrim;
        AmountFormatMinimumConfidence = amountFormatMinimumConfidence;
        AmountCharacters = amountCharacters;
        TimeCharacters = timeCharacters;
        PaymentCharacters = paymentCharacters;
        RecipientCharacters = recipientCharacters;
        PaymentBankPrefixClasses = paymentBankPrefixClasses;
        StatusClasses = statusClasses;
        StatusTextCharacters = statusTextCharacters;
        OutputNames = architectureVersion == 13 ? V13OutputNames : V12OutputNames;
        OutputShapes = outputShapes;
        StatusRuntimePolicy = statusRuntimePolicy;
        StatusReviewValue = statusReviewValue;
        TextRuntimePolicy = textRuntimePolicy;
        TextReviewValue = textReviewValue;
    }

    public string ModelPath { get; }
    public int ArchitectureVersion { get; }
    public string LabelsPath { get; }
    public string ContractPath { get; }
    public string ModelSha256 { get; }
    public string LabelsSha256 { get; }
    public string ContractSha256 { get; }
    public int FieldHeight { get; }
    public int FieldWidth { get; }
    public int RecipientHeight { get; }
    public int RecipientWidth { get; }
    public double RecipientLeftTrim { get; }
    public float AmountFormatMinimumConfidence { get; }
    public IReadOnlyList<string> AmountCharacters { get; }
    public IReadOnlyList<string> TimeCharacters { get; }
    public IReadOnlyList<string> PaymentCharacters { get; }
    public IReadOnlyList<string> RecipientCharacters { get; }
    public IReadOnlyList<string> PaymentBankPrefixClasses { get; }
    public IReadOnlyList<string> StatusClasses { get; }
    public IReadOnlyList<string>? StatusTextCharacters { get; }
    public bool HasStatusTextCtc => StatusTextCharacters is not null;
    public IReadOnlyList<string> OutputNames { get; }
    public IReadOnlyDictionary<string, int[]> OutputShapes { get; }
    public string StatusRuntimePolicy { get; }
    public string? StatusReviewValue { get; }
    public string TextRuntimePolicy { get; }
    public string TextReviewValue { get; }

    public static UnifiedOcrBundle LoadAndVerify(string modelPath)
    {
        var fullModelPath = Path.GetFullPath(modelPath);
        if (!File.Exists(fullModelPath))
        {
            throw new UsageException($"Unified OCR ONNX model not found: {fullModelPath}");
        }

        var labelsPath = Path.ChangeExtension(fullModelPath, ".labels.json");
        var contractPath = Path.ChangeExtension(fullModelPath, ".contract.json");
        if (!File.Exists(labelsPath) || !File.Exists(contractPath))
        {
            throw new UsageException(
                $"Unified OCR delivery needs {Path.GetFileName(labelsPath)} and {Path.GetFileName(contractPath)} beside {Path.GetFileName(fullModelPath)}");
        }

        using var labelsDocument = ReadJsonObject(labelsPath, "Unified OCR labels");
        using var contractDocument = ReadJsonObject(contractPath, "Unified OCR contract");
        var labels = labelsDocument.RootElement;
        var contract = contractDocument.RootElement;

        RequireInteger(contract, "schema_version", contractPath, 1);
        var artifactKind = RequireStringValue(contract, "kind", contractPath);
        RequireString(contract, "onnx_file", contractPath, Path.GetFileName(fullModelPath));
        RequireString(contract, "labels_file", contractPath, Path.GetFileName(labelsPath));
        var modelSha256 = Sha256(fullModelPath);
        var labelsSha256 = Sha256(labelsPath);
        var contractSha256 = Sha256(contractPath);
        RequireHash(contract, "onnx_sha256", modelSha256, contractPath);
        RequireHash(contract, "labels_sha256", labelsSha256, contractPath);
        RequireStringArray(contract, "slot_order", contractPath, SlotOrder);

        var model = RequireObject(contract, "model", contractPath);
        var architectureVersion = RequirePositiveInteger(model, "architecture_version", contractPath, minimum: 1);
        var expectedKind = architectureVersion switch
        {
            12 => KindV12,
            13 => KindV13,
            _ => throw ContractError(contractPath, "model.architecture_version must be 12 or 13"),
        };
        if (!string.Equals(artifactKind, expectedKind, StringComparison.Ordinal))
        {
            throw ContractError(
                contractPath,
                $"kind must be '{expectedKind}' for architecture_version={architectureVersion}, found '{artifactKind}'");
        }
        var fieldHeight = RequirePositiveInteger(model, "image_height", contractPath, minimum: 16);
        var fieldWidth = RequirePositiveInteger(model, "image_width", contractPath, minimum: 64);
        var recipientHeight = RequirePositiveInteger(model, "recipient_input_height", contractPath, minimum: 16);
        var recipientWidth = RequirePositiveInteger(model, "recipient_input_width", contractPath, minimum: 64);
        if (fieldWidth % 4 != 0 || recipientWidth % 4 != 0)
        {
            throw ContractError(contractPath, "image_width and recipient_input_width must both be static multiples of 4");
        }
        var recipientLeftTrim = RequireFiniteDouble(model, "recipient_value_left_trim", contractPath, 0.0, double.BitDecrement(1.0));
        var amountFormatMinimumConfidence = RequireFiniteFloat(model, "amount_format_min_confidence", contractPath, 0.0f, 1.0f);

        RequireInteger(labels, "schema_version", labelsPath, 1);
        RequireInteger(labels, "amount_blank_index", labelsPath, 0);
        RequireInteger(labels, "time_blank_index", labelsPath, 0);
        RequireInteger(labels, "payment_blank_index", labelsPath, 0);
        RequireInteger(labels, "recipient_blank_index", labelsPath, 0);
        var amountCharacters = RequireStringList(labels, "amount_characters", labelsPath, oneRuneEach: true);
        var timeCharacters = RequireStringList(labels, "time_characters", labelsPath, oneRuneEach: true);
        var paymentCharacters = RequireStringList(labels, "payment_characters", labelsPath, oneRuneEach: true);
        var recipientCharacters = RequireStringList(labels, "recipient_characters", labelsPath, oneRuneEach: true);
        RequireSequence(amountCharacters, ExpectedAmountCharacters, labelsPath, "amount_characters");
        RequireSequence(timeCharacters, ExpectedTimeCharacters, labelsPath, "time_characters");
        RequireUnique(paymentCharacters, labelsPath, "payment_characters");
        RequireUnique(recipientCharacters, labelsPath, "recipient_characters");
        RequireSorted(recipientCharacters, labelsPath, "recipient_characters");
        RequireHash(labels, "payment_charset_sha256", Sha256Utf8(string.Concat(paymentCharacters)), labelsPath);
        RequireHash(labels, "recipient_charset_sha256", Sha256Utf8(string.Concat(recipientCharacters)), labelsPath);
        var statusClasses = RequireStringList(labels, "status_classes", labelsPath, oneRuneEach: false);
        RequireSequence(statusClasses, ExpectedStatusClasses, labelsPath, "status_classes");
        IReadOnlyList<string>? statusTextCharacters = null;
        if (architectureVersion == 13)
        {
            RequireInteger(labels, "status_text_blank_index", labelsPath, 0);
            statusTextCharacters = RequireStringList(
                labels,
                "status_text_characters",
                labelsPath,
                oneRuneEach: true);
            RequireUnique(statusTextCharacters, labelsPath, "status_text_characters");
            RequireSorted(statusTextCharacters, labelsPath, "status_text_characters");
            RequireString(labels, "status_text_charset_source", labelsPath, StatusTextCharsetSource);
            RequireString(contract, "status_text_charset_source", contractPath, StatusTextCharsetSource);
            RequireString(labels, "status_text_target", labelsPath, StatusTextTarget);
            RequireString(contract, "status_text_target", contractPath, StatusTextTarget);
            RequireString(labels, "status_text_runtime_policy", labelsPath, StatusTextRuntimePolicy);
            RequireString(contract, "status_text_runtime_policy", contractPath, StatusTextRuntimePolicy);
            var statusTextCharsetSha256 = Sha256Utf8(string.Concat(statusTextCharacters));
            RequireHash(labels, "status_text_charset_sha256", statusTextCharsetSha256, labelsPath);
            RequireHash(contract, "status_text_charset_sha256", statusTextCharsetSha256, contractPath);
        }
        else if (labels.TryGetProperty("status_text_blank_index", out _)
            || labels.TryGetProperty("status_text_characters", out _)
            || labels.TryGetProperty("status_text_charset_source", out _)
            || labels.TryGetProperty("status_text_charset_sha256", out _)
            || labels.TryGetProperty("status_text_target", out _)
            || labels.TryGetProperty("status_text_runtime_policy", out _)
            || contract.TryGetProperty("status_text_charset_source", out _)
            || contract.TryGetProperty("status_text_charset_sha256", out _)
            || contract.TryGetProperty("status_text_target", out _)
            || contract.TryGetProperty("status_text_runtime_policy", out _))
        {
            throw ContractError(labelsPath, "v12 artifacts must not declare the v13 status-text CTC vocabulary");
        }

        var paymentBankPrefixClasses = RequireStringList(labels, "payment_bank_prefix_classes", labelsPath, oneRuneEach: false);
        if (paymentBankPrefixClasses.Count < 2 || !string.Equals(paymentBankPrefixClasses[0], "__other__", StringComparison.Ordinal))
        {
            throw ContractError(labelsPath, "payment_bank_prefix_classes must start with __other__ and include at least one bank class");
        }
        RequireUnique(paymentBankPrefixClasses, labelsPath, "payment_bank_prefix_classes");
        RequireSorted(paymentBankPrefixClasses.Skip(1).ToArray(), labelsPath, "payment_bank_prefix_classes after __other__");
        RequireStringArray(contract, "payment_bank_prefix_classes", contractPath, paymentBankPrefixClasses);

        RequireString(labels, "recipient_charset_source", labelsPath, RecipientCharsetSource);
        RequireString(labels, "recipient_target", labelsPath, RecipientTarget);
        RequireString(contract, "recipient_charset_source", contractPath, RecipientCharsetSource);
        RequireString(contract, "recipient_target", contractPath, RecipientTarget);
        RequireEqualJsonValue(contract, "recipient_oov_by_split", labels, "recipient_oov_by_split", contractPath, labelsPath);
        RequireRecipientOovAudit(labels, labelsPath);

        RequireString(labels, "recipient_input_preprocess", labelsPath, RecipientPreprocess);
        RequireString(contract, "recipient_input_preprocess", contractPath, RecipientPreprocess);
        RequireDoubleEqual(labels, "recipient_value_left_trim", recipientLeftTrim, labelsPath);
        RequireDoubleEqual(contract, "recipient_value_left_trim", recipientLeftTrim, contractPath);
        RequireIntArray(labels, "recipient_input_shape", labelsPath, [1, 1, recipientHeight, recipientWidth]);
        RequireIntArray(contract, "recipient_input_shape", contractPath, [1, 1, recipientHeight, recipientWidth]);
        RequireString(labels, "recipient_input_name", labelsPath, "recipient_value_image");
        RequireString(contract, "recipient_input_name", contractPath, "recipient_value_image");

        var textPolicy = RequireObject(contract, "text_delivery_policy", contractPath);
        RequireString(textPolicy, "runtime_policy", contractPath, TextReviewPolicy);
        RequireString(textPolicy, "review_value", contractPath, ReviewValue);
        var statusPolicy = RequireObject(contract, "status_head_policy", contractPath);
        var statusRuntimePolicy = RequireStringValue(statusPolicy, "runtime_policy", contractPath);
        if (statusRuntimePolicy is not ("review_only" or "classify"))
        {
            throw ContractError(contractPath, "status_head_policy.runtime_policy must be review_only or classify");
        }
        if (architectureVersion == 13 && statusRuntimePolicy != "review_only")
        {
            throw ContractError(
                contractPath,
                "v13 status_logits must remain review_only because visible status text supersedes the legacy classifier");
        }

        var primaryInput = RequireObject(contract, "input", contractPath);
        ValidateInput(primaryInput, "field_images", [5, 1, fieldHeight, fieldWidth], contractPath);
        var inputs = RequireArray(contract, "inputs", contractPath);
        if (inputs.GetArrayLength() != 2 || !JsonDeepEquals(inputs[0], primaryInput))
        {
            throw ContractError(contractPath, $"v{architectureVersion} contract must declare exactly two ordered ONNX inputs with input as the first entry");
        }
        var recipientInput = inputs[1];
        if (recipientInput.ValueKind != JsonValueKind.Object)
        {
            throw ContractError(contractPath, $"v{architectureVersion} recipient input contract must be an object");
        }
        ValidateInput(recipientInput, "recipient_value_image", [1, 1, recipientHeight, recipientWidth], contractPath);
        RequireString(recipientInput, "absent_slot_policy", contractPath, WhitePlaceholderPolicy);

        var structuredDecoder = RequireObject(labels, "structured_decoder", labelsPath);
        ValidateStructuredDecoder(structuredDecoder, amountFormatMinimumConfidence, labelsPath);

        var outputs = RequireObject(contract, "outputs", contractPath);
        var outputNames = outputs.EnumerateObject().Select(value => value.Name).OrderBy(value => value, StringComparer.Ordinal).ToArray();
        var versionedOutputNames = architectureVersion == 13 ? V13OutputNames : V12OutputNames;
        var expectedOutputNames = versionedOutputNames.OrderBy(value => value, StringComparer.Ordinal).ToArray();
        RequireSequence(outputNames, expectedOutputNames, contractPath, "outputs");
        var expectedOutputShapes = new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["amount_logits"] = [fieldWidth / 4, amountCharacters.Count + 1],
            ["time_logits"] = [fieldWidth / 4, timeCharacters.Count + 1],
            ["payment_logits"] = [fieldWidth / 4, paymentCharacters.Count + 1],
            ["status_logits"] = [statusClasses.Count],
            ["amount_currency_style_logits"] = [ExpectedAmountCurrencyStyles.Length],
            ["amount_grouped_thousands_logits"] = [ExpectedAmountGroupingStyles.Length],
            ["amount_sign_position_logits"] = [ExpectedAmountSignPositions.Length],
            ["time_format_logits"] = [ExpectedTimeFormats.Length],
            ["time_digit_logits"] = [14, 10],
            ["payment_prefix_logits"] = [fieldWidth / 4, paymentCharacters.Count + 1],
            ["payment_bank_prefix_logits"] = [paymentBankPrefixClasses.Count],
            ["payment_tail_digit_logits"] = [4, 10],
            ["payment_structure_logits"] = [ExpectedPaymentStructureClasses.Length],
            ["payment_parentheses_logits"] = [ExpectedPaymentParenthesisClasses.Length],
            ["recipient_logits"] = [recipientWidth / 4, recipientCharacters.Count + 1],
        };
        if (architectureVersion == 13)
        {
            if (statusTextCharacters is null)
            {
                throw new InvalidOperationException("Verified v13 status-text characters are missing");
            }
            expectedOutputShapes[StatusTextOutputName] = [fieldWidth / 4, statusTextCharacters.Count + 1];
        }
        foreach (var (name, shape) in expectedOutputShapes)
        {
            var output = RequireObject(outputs, name, contractPath);
            RequireIntArray(output, "shape", contractPath, shape);
            if (name is "amount_logits" or "time_logits" or "payment_logits" or "payment_prefix_logits" or "recipient_logits" or StatusTextOutputName)
            {
                RequireInteger(output, "blank_index", contractPath, 0);
            }
            else if (output.TryGetProperty("blank_index", out _))
            {
                throw ContractError(contractPath, $"structured output {name} must not declare blank_index");
            }
        }

        var recipientOutput = RequireObject(outputs, "recipient_logits", contractPath);
        RequireString(recipientOutput, "characters", contractPath, "recipient_characters");
        RequireString(recipientOutput, "target", contractPath, RecipientTarget);
        RequireString(recipientOutput, "runtime_policy", contractPath, "review_only");
        RequireString(recipientOutput, "input_preprocess", contractPath, RecipientPreprocess);
        RequireString(recipientOutput, "input_name", contractPath, "recipient_value_image");
        RequireString(recipientOutput, "horizontal_alignment", contractPath, "center");
        RequireDoubleEqual(recipientOutput, "left_trim_fraction", recipientLeftTrim, contractPath);

        var statusOutput = RequireObject(outputs, "status_logits", contractPath);
        RequireString(statusOutput, "runtime_policy", contractPath, statusRuntimePolicy);
        var statusReviewValue = statusRuntimePolicy == "review_only"
            ? RequireStringValue(statusOutput, "review_value", contractPath)
            : null;
        if (statusRuntimePolicy == "review_only" && !string.Equals(statusReviewValue, ReviewValue, StringComparison.Ordinal))
        {
            throw ContractError(contractPath, "review-only status output must use review_value=review");
        }

        if (architectureVersion == 13)
        {
            var statusTextOutput = RequireObject(outputs, StatusTextOutputName, contractPath);
            RequireString(statusTextOutput, "layout", contractPath, "[time,class]");
            RequireString(statusTextOutput, "decoder", contractPath, "ctc_greedy");
            RequireString(statusTextOutput, "characters", contractPath, "status_text_characters");
            RequireString(statusTextOutput, "target", contractPath, StatusTextTarget);
            RequireString(statusTextOutput, "runtime_policy", contractPath, StatusTextRuntimePolicy);
            RequireString(statusTextOutput, "review_value", contractPath, ReviewValue);
            RequireString(statusTextOutput, "normalizer", contractPath, StatusTextNormalizer);
        }

        return new UnifiedOcrBundle(
            fullModelPath,
            architectureVersion,
            labelsPath,
            contractPath,
            modelSha256,
            labelsSha256,
            contractSha256,
            fieldHeight,
            fieldWidth,
            recipientHeight,
            recipientWidth,
            recipientLeftTrim,
            amountFormatMinimumConfidence,
            amountCharacters,
            timeCharacters,
            paymentCharacters,
            recipientCharacters,
            paymentBankPrefixClasses,
            statusClasses,
            statusTextCharacters,
            expectedOutputShapes,
            statusRuntimePolicy,
            statusReviewValue,
            RequireStringValue(textPolicy, "runtime_policy", contractPath),
            RequireStringValue(textPolicy, "review_value", contractPath));
    }

    private static void ValidateStructuredDecoder(JsonElement decoder, float expectedThreshold, string path)
    {
        RequireInteger(decoder, "schema_version", path, 1);
        RequireString(decoder, "amount_visible_format", path, "visible_cny_amount_format_v8");
        RequireStringArray(decoder, "amount_currency_style_classes", path, ExpectedAmountCurrencyStyles);
        RequireStringArray(decoder, "amount_grouped_thousands_classes", path, ExpectedAmountGroupingStyles);
        RequireStringArray(decoder, "amount_sign_position_classes", path, ExpectedAmountSignPositions);
        RequireFloatEqual(decoder, "amount_format_min_confidence", expectedThreshold, path);
        RequireString(decoder, "amount_rendering", path, "canonical_amount_ctc + finite_display_grammar_only_when_all_components_confident");
        RequireInteger(decoder, "time_digit_slots", path, 14);
        RequireStringArray(decoder, "time_display_format_classes", path, ExpectedTimeFormats);
        RequireString(decoder, "time_visible_format", path, "visible_clock_or_datetime_strict_v6");
        RequireInteger(decoder, "payment_tail_digit_slots", path, 4);
        RequireStringArray(decoder, "payment_structure_classes", path, ExpectedPaymentStructureClasses);
        RequireStringArray(decoder, "payment_parentheses_classes", path, ExpectedPaymentParenthesisClasses);
        RequireString(decoder, "payment_bank_prefix_format", path, "visible_payment_bank_prefix_v6");
        RequireString(decoder, "payment_bank_prefix_other_class", path, "__other__");
    }

    private static void ValidateInput(JsonElement input, string name, int[] shape, string path)
    {
        RequireString(input, "name", path, name);
        RequireString(input, "dtype", path, "float32");
        RequireIntArray(input, "shape", path, shape);
        RequireString(input, "absent_slot_policy", path, WhitePlaceholderPolicy);
    }

    private static void RequireRecipientOovAudit(JsonElement labels, string path)
    {
        var audit = RequireObject(labels, "recipient_oov_by_split", path);
        var names = audit.EnumerateObject().Select(item => item.Name).OrderBy(item => item, StringComparer.Ordinal).ToArray();
        RequireSequence(names, ["test", "train", "val"], path, "recipient_oov_by_split keys");
        foreach (var split in new[] { "train", "val", "test" })
        {
            var row = RequireObject(audit, split, path);
            var records = RequireNonNegativeInteger(row, "records", path);
            var oovRecords = RequireNonNegativeInteger(row, "oov_records", path);
            if (oovRecords > records || (split == "train" && oovRecords != 0))
            {
                throw ContractError(path, "recipient OOV audit is invalid");
            }
        }
    }

    private static JsonDocument ReadJsonObject(string path, string description)
    {
        try
        {
            var document = JsonDocument.Parse(File.ReadAllBytes(path));
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                document.Dispose();
                throw ContractError(path, $"{description} must be a JSON object");
            }
            return document;
        }
        catch (JsonException error)
        {
            throw new UsageException($"{path}: invalid JSON: {error.Message}");
        }
    }

    private static JsonElement RequireObject(JsonElement source, string name, string path)
    {
        if (!source.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Object)
        {
            throw ContractError(path, $"missing or invalid object {name}");
        }
        return value;
    }

    private static JsonElement RequireArray(JsonElement source, string name, string path)
    {
        if (!source.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array)
        {
            throw ContractError(path, $"missing or invalid array {name}");
        }
        return value;
    }

    private static string RequireStringValue(JsonElement source, string name, string path)
    {
        if (!source.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw ContractError(path, $"missing or invalid string {name}");
        }
        return value.GetString()!;
    }

    private static void RequireString(JsonElement source, string name, string path, string expected)
    {
        var actual = RequireStringValue(source, name, path);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw ContractError(path, $"{name} must be '{expected}', found '{actual}'");
        }
    }

    private static void RequireHash(JsonElement source, string name, string expected, string path)
    {
        var actual = RequireStringValue(source, name, path);
        if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
        {
            throw ContractError(path, $"{name} does not match the adjacent delivery artifact");
        }
    }

    private static void RequireInteger(JsonElement source, string name, string path, int expected)
    {
        if (!source.TryGetProperty(name, out var value) || !value.TryGetInt32(out var actual) || actual != expected)
        {
            throw ContractError(path, $"{name} must be integer {expected}");
        }
    }

    private static int RequirePositiveInteger(JsonElement source, string name, string path, int minimum)
    {
        if (!source.TryGetProperty(name, out var value) || !value.TryGetInt32(out var actual) || actual < minimum)
        {
            throw ContractError(path, $"{name} must be an integer >= {minimum}");
        }
        return actual;
    }

    private static int RequireNonNegativeInteger(JsonElement source, string name, string path)
    {
        if (!source.TryGetProperty(name, out var value) || !value.TryGetInt32(out var actual) || actual < 0)
        {
            throw ContractError(path, $"{name} must be a non-negative integer");
        }
        return actual;
    }

    private static float RequireFiniteFloat(JsonElement source, string name, string path, float minimum, float maximum)
    {
        if (!source.TryGetProperty(name, out var value) || !value.TryGetSingle(out var actual) || !float.IsFinite(actual) || actual < minimum || actual > maximum)
        {
            throw ContractError(path, $"{name} must be a finite number in [{minimum}, {maximum}]");
        }
        return actual;
    }

    private static double RequireFiniteDouble(JsonElement source, string name, string path, double minimum, double maximum)
    {
        if (!source.TryGetProperty(name, out var value) || !value.TryGetDouble(out var actual) || !double.IsFinite(actual) || actual < minimum || actual > maximum)
        {
            throw ContractError(path, $"{name} must be a finite number in [{minimum}, {maximum}]");
        }
        return actual;
    }

    private static void RequireFloatEqual(JsonElement source, string name, float expected, string path)
    {
        var actual = RequireFiniteFloat(source, name, path, float.NegativeInfinity, float.PositiveInfinity);
        if (MathF.Abs(actual - expected) > 1e-7f)
        {
            throw ContractError(path, $"{name} does not match the model configuration");
        }
    }

    private static void RequireDoubleEqual(JsonElement source, string name, double expected, string path)
    {
        var actual = RequireFiniteDouble(source, name, path, double.NegativeInfinity, double.PositiveInfinity);
        if (Math.Abs(actual - expected) > 1e-12)
        {
            throw ContractError(path, $"{name} does not match the model configuration");
        }
    }

    private static IReadOnlyList<string> RequireStringList(JsonElement source, string name, string path, bool oneRuneEach)
    {
        var array = RequireArray(source, name, path);
        var values = new List<string>(array.GetArrayLength());
        foreach (var value in array.EnumerateArray())
        {
            if (value.ValueKind != JsonValueKind.String || string.IsNullOrEmpty(value.GetString()))
            {
                throw ContractError(path, $"{name} must contain non-empty strings");
            }
            var text = value.GetString()!;
            if (oneRuneEach && text.EnumerateRunes().Count() != 1)
            {
                throw ContractError(path, $"{name} must contain one Unicode code point per entry");
            }
            values.Add(text);
        }
        if (values.Count == 0)
        {
            throw ContractError(path, $"{name} must not be empty");
        }
        return values;
    }

    private static void RequireStringArray(JsonElement source, string name, string path, IReadOnlyList<string> expected)
    {
        RequireSequence(RequireStringList(source, name, path, oneRuneEach: false), expected, path, name);
    }

    private static void RequireIntArray(JsonElement source, string name, string path, IReadOnlyList<int> expected)
    {
        var array = RequireArray(source, name, path);
        if (array.GetArrayLength() != expected.Count)
        {
            throw ContractError(path, $"{name} has an invalid static shape");
        }
        for (var index = 0; index < expected.Count; index++)
        {
            if (!array[index].TryGetInt32(out var actual) || actual != expected[index])
            {
                throw ContractError(path, $"{name} has an invalid static shape");
            }
        }
    }

    private static void RequireUnique(IReadOnlyList<string> values, string path, string name)
    {
        if (values.Distinct(StringComparer.Ordinal).Count() != values.Count)
        {
            throw ContractError(path, $"{name} must not contain duplicates");
        }
    }

    private static void RequireSorted(IReadOnlyList<string> values, string path, string name)
    {
        if (!values.SequenceEqual(values.OrderBy(value => value, StringComparer.Ordinal), StringComparer.Ordinal))
        {
            throw ContractError(path, $"{name} must be ordinal-sorted");
        }
    }

    private static void RequireSequence(IEnumerable<string> actual, IEnumerable<string> expected, string path, string name)
    {
        if (!actual.SequenceEqual(expected, StringComparer.Ordinal))
        {
            throw ContractError(path, $"{name} is unsupported");
        }
    }

    private static void RequireEqualJsonValue(JsonElement left, string leftName, JsonElement right, string rightName, string leftPath, string rightPath)
    {
        if (!left.TryGetProperty(leftName, out var leftValue)
            || !right.TryGetProperty(rightName, out var rightValue)
            || !JsonDeepEquals(leftValue, rightValue))
        {
            throw ContractError(leftPath, $"{leftName} must match {rightPath}:{rightName}");
        }
    }

    // JsonElement.DeepEquals is not available in the .NET 8 runtime used by
    // the production host. Keep contract comparison semantic (object property
    // order does not matter) without raising the target framework.
    private static bool JsonDeepEquals(JsonElement left, JsonElement right)
    {
        if (left.ValueKind != right.ValueKind)
        {
            return false;
        }
        switch (left.ValueKind)
        {
            case JsonValueKind.Object:
            {
                var leftProperties = left.EnumerateObject()
                    .OrderBy(property => property.Name, StringComparer.Ordinal)
                    .ToArray();
                var rightProperties = right.EnumerateObject()
                    .OrderBy(property => property.Name, StringComparer.Ordinal)
                    .ToArray();
                return leftProperties.Length == rightProperties.Length
                    && leftProperties.Zip(rightProperties).All(pair =>
                        pair.First.Name == pair.Second.Name
                        && JsonDeepEquals(pair.First.Value, pair.Second.Value));
            }
            case JsonValueKind.Array:
            {
                var leftItems = left.EnumerateArray().ToArray();
                var rightItems = right.EnumerateArray().ToArray();
                return leftItems.Length == rightItems.Length
                    && leftItems.Zip(rightItems).All(pair => JsonDeepEquals(pair.First, pair.Second));
            }
            case JsonValueKind.String:
                return left.GetString() == right.GetString();
            case JsonValueKind.Number:
                return left.TryGetDecimal(out var leftDecimal)
                    && right.TryGetDecimal(out var rightDecimal)
                        ? leftDecimal == rightDecimal
                        : left.GetRawText() == right.GetRawText();
            case JsonValueKind.True:
            case JsonValueKind.False:
                return left.GetBoolean() == right.GetBoolean();
            case JsonValueKind.Null:
            case JsonValueKind.Undefined:
                return true;
            default:
                return left.GetRawText() == right.GetRawText();
        }
    }

    private static UsageException ContractError(string path, string message) => new($"Unified OCR contract rejected ({path}): {message}");

    private static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        using var algorithm = SHA256.Create();
        return Convert.ToHexString(algorithm.ComputeHash(stream)).ToLowerInvariant();
    }

    private static string Sha256Utf8(string text)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
