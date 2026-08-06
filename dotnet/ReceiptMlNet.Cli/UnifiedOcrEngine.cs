using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Diagnostic result emitted by the v12 unified OCR reader.  The delivery
/// value is deliberately separate from the candidate: current v12 text
/// contracts are review-only until a human-truth calibration has passed.
/// </summary>
internal sealed record UnifiedOcrCandidate(
    string Candidate,
    float Confidence,
    string CtcCandidate,
    float CtcConfidence,
    string? StructuredCandidate,
    float? StructuredConfidence,
    string DeliveryValue);

internal sealed record UnifiedOcrReadResult(
    IReadOnlyDictionary<string, UnifiedOcrCandidate> Candidates,
    string? StatusCandidate,
    float? StatusConfidence,
    string StatusDeliveryValue,
    string StatusRuntimePolicy,
    string TextDeliveryValue,
    string TextRuntimePolicy);

/// <summary>
/// Observability-only timing for the unified reader.  The three measurements
/// bracket the existing input construction, single ONNX Runtime Run call and
/// output materialisation/decoding respectively.
/// </summary>
internal sealed record UnifiedOcrStageLatency(
    double PreprocessMs,
    double InferenceMs,
    double PostprocessMs);

/// <summary>
/// One-session implementation of architecture-v12 unified receipt OCR.
/// Every receipt produces exactly one ONNX Runtime Run call with the two
/// fixed contract inputs.  The model, labels and contract are verified by
/// <see cref="UnifiedOcrBundle"/> before this session is opened.
/// </summary>
internal sealed class UnifiedOcrEngine : IDisposable
{
    private const string FieldImagesInput = "field_images";
    private const string RecipientValueInput = "recipient_value_image";

    private static readonly string[] InputNames = [FieldImagesInput, RecipientValueInput];
    private static readonly string[] AmountCurrencyStyles = ["none", "yen", "yen_space", "fullwidth_yen", "fullwidth_yen_space"];
    private static readonly string[] AmountGroupingStyles = ["ungrouped", "grouped_thousands"];
    private static readonly string[] AmountSignPositions = ["none", "before_currency_or_number", "after_currency"];
    private static readonly string[] TimeFormats =
    [
        "clock_h_mm",
        "clock_hh_mm",
        "clock_h_mm_ss",
        "clock_hh_mm_ss",
        "date_ymd_hh_mm",
        "date_ymd_hh_mm_ss",
    ];

    private readonly UnifiedOcrBundle _bundle;
    private readonly InferenceSession _session;

    public UnifiedOcrEngine(UnifiedOcrBundle bundle, DeviceSetting requestedDevice)
    {
        _bundle = bundle;
        _session = CreateSession(bundle.ModelPath, requestedDevice, out var provider);
        ExecutionProvider = provider;
        try
        {
            VerifyRuntimeAbi();
        }
        catch
        {
            _session.Dispose();
            throw;
        }
    }

    public string ExecutionProvider { get; }

    /// <summary>
    /// Builds the frozen two-input ABI and invokes the model once.  Missing
    /// detection/crop slots stay all-white and are intentionally not decoded.
    /// </summary>
    public UnifiedOcrReadResult RecognizeReceipt(
        Image<Rgb24> source,
        IReadOnlyList<DetectionResult> detections,
        out UnifiedOcrStageLatency stageLatency)
    {
        var stageStopwatch = System.Diagnostics.Stopwatch.StartNew();
        var byLabel = detections.ToDictionary(item => item.Label, StringComparer.Ordinal);
        var fieldValues = Enumerable.Repeat(1.0f, checked(5 * _bundle.FieldHeight * _bundle.FieldWidth)).ToArray();
        var recipientValues = Enumerable.Repeat(1.0f, checked(_bundle.RecipientHeight * _bundle.RecipientWidth)).ToArray();

        var readable = new HashSet<string>(StringComparer.Ordinal);
        WriteField(byLabel.GetValueOrDefault("amount"), 0, rightAlign: true, source, fieldValues, readable, "amount");
        WriteField(byLabel.GetValueOrDefault("time"), 1, rightAlign: true, source, fieldValues, readable, "time");
        WriteField(byLabel.GetValueOrDefault("transfer_status"), 2, rightAlign: false, source, fieldValues, readable, "transfer_status");
        WriteField(byLabel.GetValueOrDefault("payment_method_field"), 3, rightAlign: true, source, fieldValues, readable, "payment_method_field");
        // Architecture v12 freezes channel 4 as white. Recipient text is read
        // only from the separate high-resolution input below.
        WriteRecipient(byLabel.GetValueOrDefault("recipient_field"), source, recipientValues, readable);

        var fieldTensor = new DenseTensor<float>(fieldValues, [5, 1, _bundle.FieldHeight, _bundle.FieldWidth]);
        var recipientTensor = new DenseTensor<float>(recipientValues, [1, 1, _bundle.RecipientHeight, _bundle.RecipientWidth]);
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor(FieldImagesInput, fieldTensor),
            NamedOnnxValue.CreateFromTensor(RecipientValueInput, recipientTensor),
        };
        var preprocessMs = StopAndReadMilliseconds(stageStopwatch);

        Dictionary<string, OrtOutput> outputs;
        double inferenceMs;
        stageStopwatch.Restart();
        using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runtimeOutputs = _session.Run(inputs))
        {
            inferenceMs = StopAndReadMilliseconds(stageStopwatch);
            stageStopwatch.Restart();
            outputs = ReadOutputs(runtimeOutputs);
        }

        var candidates = new Dictionary<string, UnifiedOcrCandidate>(StringComparer.Ordinal);
        if (readable.Contains("amount"))
        {
            candidates["amount"] = DecodeAmount(outputs);
        }
        if (readable.Contains("time"))
        {
            candidates["time"] = DecodeTime(outputs);
        }
        if (readable.Contains("payment_method_field"))
        {
            candidates["payment_method_field"] = DecodeCtcCandidate(
                outputs["payment_logits"], _bundle.PaymentCharacters);
        }
        if (readable.Contains("recipient_field"))
        {
            candidates["recipient_field"] = DecodeCtcCandidate(
                outputs["recipient_logits"], _bundle.RecipientCharacters);
        }

        string? statusCandidate = null;
        float? statusConfidence = null;
        if (readable.Contains("transfer_status"))
        {
            var status = DecodeClass(outputs["status_logits"]);
            statusCandidate = _bundle.StatusClasses[status.Index];
            statusConfidence = status.Confidence;
        }

        var result = new UnifiedOcrReadResult(
            candidates,
            statusCandidate,
            statusConfidence,
            _bundle.StatusRuntimePolicy == "classify" ? statusCandidate ?? "review" : _bundle.StatusReviewValue ?? "review",
            _bundle.StatusRuntimePolicy,
            _bundle.TextReviewValue,
            _bundle.TextRuntimePolicy);
        var postprocessMs = StopAndReadMilliseconds(stageStopwatch);
        stageLatency = new UnifiedOcrStageLatency(preprocessMs, inferenceMs, postprocessMs);
        return result;
    }

    private static double StopAndReadMilliseconds(System.Diagnostics.Stopwatch stopwatch)
    {
        stopwatch.Stop();
        return Math.Round(stopwatch.Elapsed.TotalMilliseconds, 4);
    }

    public void Dispose() => _session.Dispose();

    private void WriteField(
        DetectionResult? detection,
        int slot,
        bool rightAlign,
        Image<Rgb24> source,
        float[] destination,
        ISet<string> readable,
        string label)
    {
        if (detection is null)
        {
            return;
        }
        using var crop = UnifiedOcrImageOps.CropFieldWithMargin(source, detection.BboxImage);
        if (crop is null)
        {
            return;
        }
        var values = UnifiedOcrImageOps.PrepareFieldTensor(crop, _bundle.FieldHeight, _bundle.FieldWidth, rightAlign);
        values.CopyTo(destination, checked(slot * _bundle.FieldHeight * _bundle.FieldWidth));
        readable.Add(label);
    }

    private void WriteRecipient(
        DetectionResult? detection,
        Image<Rgb24> source,
        float[] destination,
        ISet<string> readable)
    {
        if (detection is null)
        {
            return;
        }
        using var crop = UnifiedOcrImageOps.CropFieldWithMargin(source, detection.BboxImage);
        if (crop is null)
        {
            return;
        }
        var values = UnifiedOcrImageOps.PrepareFieldTensor(
            crop,
            _bundle.RecipientHeight,
            _bundle.RecipientWidth,
            rightAlign: false,
            _bundle.RecipientLeftTrim);
        values.CopyTo(destination, 0);
        readable.Add("recipient_field");
    }

    private Dictionary<string, OrtOutput> ReadOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runtimeOutputs)
    {
        var names = runtimeOutputs.Select(output => output.Name).ToArray();
        if (!HasExactNames(names, UnifiedOcrBundle.OutputNames))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime outputs differ from its v12 contract: [{string.Join(',', names)}]");
        }

        var output = new Dictionary<string, OrtOutput>(StringComparer.Ordinal);
        foreach (var namedValue in runtimeOutputs)
        {
            var tensor = namedValue.AsTensor<float>();
            var shape = tensor.Dimensions.ToArray();
            if (!_bundle.OutputShapes.TryGetValue(namedValue.Name, out var expected)
                || !shape.SequenceEqual(expected))
            {
                throw new InvalidOperationException(
                    $"Unified OCR runtime output {namedValue.Name} has invalid static shape [{string.Join(',', shape)}]");
            }
            output.Add(namedValue.Name, new OrtOutput(tensor.ToArray(), shape));
        }
        return output;
    }

    private UnifiedOcrCandidate DecodeAmount(IReadOnlyDictionary<string, OrtOutput> outputs)
    {
        var ctc = DecodeCtc(outputs["amount_logits"], _bundle.AmountCharacters);
        var structured = TryDecodeStructuredAmount(
            ctc,
            outputs["amount_currency_style_logits"],
            outputs["amount_grouped_thousands_logits"],
            outputs["amount_sign_position_logits"]);
        var candidate = structured?.Text ?? ctc.Text;
        var confidence = structured?.Confidence ?? ctc.Confidence;
        return new UnifiedOcrCandidate(
            candidate,
            confidence,
            ctc.Text,
            ctc.Confidence,
            structured?.Text,
            structured?.Confidence,
            _bundle.TextReviewValue);
    }

    private UnifiedOcrCandidate DecodeTime(IReadOnlyDictionary<string, OrtOutput> outputs)
    {
        var ctc = DecodeCtc(outputs["time_logits"], _bundle.TimeCharacters);
        var structured = TryDecodeStructuredTime(outputs["time_format_logits"], outputs["time_digit_logits"]);
        var candidate = structured?.Text ?? ctc.Text;
        var confidence = structured?.Confidence ?? ctc.Confidence;
        return new UnifiedOcrCandidate(
            candidate,
            confidence,
            ctc.Text,
            ctc.Confidence,
            structured?.Text,
            structured?.Confidence,
            _bundle.TextReviewValue);
    }

    private UnifiedOcrCandidate DecodeCtcCandidate(OrtOutput output, IReadOnlyList<string> characters)
    {
        var ctc = DecodeCtc(output, characters);
        return new UnifiedOcrCandidate(
            ctc.Text,
            ctc.Confidence,
            ctc.Text,
            ctc.Confidence,
            null,
            null,
            _bundle.TextReviewValue);
    }

    private CtcRead DecodeCtc(OrtOutput output, IReadOnlyList<string> characters)
    {
        if (output.Shape.Length != 2 || output.Shape[1] != characters.Count + 1)
        {
            throw new InvalidOperationException("Unified OCR CTC tensor differs from the verified character dictionary");
        }
        var text = new System.Text.StringBuilder();
        var scores = new List<float>();
        var previous = -1;
        for (var time = 0; time < output.Shape[0]; time++)
        {
            var decoded = ArgMax(output.Values, checked(time * output.Shape[1]), output.Shape[1]);
            if (decoded.Index != 0 && decoded.Index != previous)
            {
                text.Append(characters[decoded.Index - 1]);
                scores.Add(decoded.Confidence);
            }
            previous = decoded.Index;
        }
        return new CtcRead(text.ToString(), scores.Count == 0 ? 0.0f : scores.Average());
    }

    private StructuredRead? TryDecodeStructuredAmount(
        CtcRead ctc,
        OrtOutput currencyOutput,
        OrtOutput groupingOutput,
        OrtOutput signOutput)
    {
        if (!TryParseCanonicalAmount(ctc.Text, out var negative, out var integer, out var cents))
        {
            return null;
        }
        var currency = DecodeClass(currencyOutput);
        var grouping = DecodeClass(groupingOutput);
        var sign = DecodeClass(signOutput);
        var currencyStyle = AmountCurrencyStyles[currency.Index];
        var grouped = integer.Length >= 4 ? AmountGroupingStyles[grouping.Index] : "ungrouped";
        var signPosition = negative && currencyStyle != "none"
            ? AmountSignPositions[sign.Index]
            : negative ? "before_currency_or_number" : "none";
        var relevant = new List<float> { currency.Confidence };
        if (integer.Length >= 4)
        {
            relevant.Add(grouping.Confidence);
        }
        if (negative && currencyStyle != "none")
        {
            relevant.Add(sign.Confidence);
        }
        var componentConfidence = relevant.Min();
        if (componentConfidence < _bundle.AmountFormatMinimumConfidence
            || !TryRenderVisibleAmount(negative, integer, cents, currencyStyle, grouped, signPosition, out var rendered))
        {
            return null;
        }
        return new StructuredRead(rendered, Math.Min(ctc.Confidence, componentConfidence));
    }

    private StructuredRead? TryDecodeStructuredTime(OrtOutput formatOutput, OrtOutput digitsOutput)
    {
        var format = DecodeClass(formatOutput);
        if (digitsOutput.Shape.Length != 2 || digitsOutput.Shape[0] != 14 || digitsOutput.Shape[1] != 10)
        {
            throw new InvalidOperationException("Unified OCR time-digit tensor has an invalid static shape");
        }
        var digits = new int[14];
        var confidence = new float[14];
        for (var index = 0; index < digits.Length; index++)
        {
            var decoded = ArgMax(digitsOutput.Values, index * 10, 10);
            digits[index] = decoded.Index;
            confidence[index] = decoded.Confidence;
        }
        var textDigits = string.Concat(digits.Select(value => value.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        var formatName = TimeFormats[format.Index];
        string candidate;
        var used = 0;
        switch (formatName)
        {
            case "clock_h_mm":
                candidate = $"{int.Parse(textDigits[..2], System.Globalization.CultureInfo.InvariantCulture)}:{textDigits.Substring(2, 2)}";
                used = 4;
                break;
            case "clock_hh_mm":
                candidate = $"{textDigits[..2]}:{textDigits.Substring(2, 2)}";
                used = 4;
                break;
            case "clock_h_mm_ss":
                candidate = $"{int.Parse(textDigits[..2], System.Globalization.CultureInfo.InvariantCulture)}:{textDigits.Substring(2, 2)}:{textDigits.Substring(4, 2)}";
                used = 6;
                break;
            case "clock_hh_mm_ss":
                candidate = $"{textDigits[..2]}:{textDigits.Substring(2, 2)}:{textDigits.Substring(4, 2)}";
                used = 6;
                break;
            case "date_ymd_hh_mm":
                candidate = $"{textDigits[..4]}-{textDigits.Substring(4, 2)}-{textDigits.Substring(6, 2)} {textDigits.Substring(8, 2)}:{textDigits.Substring(10, 2)}";
                used = 12;
                break;
            case "date_ymd_hh_mm_ss":
                candidate = $"{textDigits[..4]}-{textDigits.Substring(4, 2)}-{textDigits.Substring(6, 2)} {textDigits.Substring(8, 2)}:{textDigits.Substring(10, 2)}:{textDigits.Substring(12, 2)}";
                used = 14;
                break;
            default:
                throw new InvalidOperationException($"Unsupported verified time format {formatName}");
        }
        if (!IsValidTimeDisplay(candidate))
        {
            return null;
        }
        var mean = (format.Confidence + confidence.Take(used).Sum()) / (used + 1);
        return new StructuredRead(candidate, mean);
    }

    private static bool TryParseCanonicalAmount(string value, out bool negative, out string integer, out string cents)
    {
        negative = false;
        integer = string.Empty;
        cents = string.Empty;
        if (string.IsNullOrEmpty(value))
        {
            return false;
        }
        var text = value;
        if (text[0] == '-')
        {
            negative = true;
            text = text[1..];
        }
        var decimalIndex = text.IndexOf('.');
        if (decimalIndex <= 0 || decimalIndex != text.LastIndexOf('.') || text.Length - decimalIndex - 1 != 2)
        {
            return false;
        }
        integer = text[..decimalIndex];
        cents = text[(decimalIndex + 1)..];
        if (integer.Length > 7 || !integer.All(char.IsAsciiDigit) || !cents.All(char.IsAsciiDigit)
            || (integer.Length > 1 && integer[0] == '0')
            || (negative && integer == "0" && cents == "00"))
        {
            return false;
        }
        return true;
    }

    private static bool TryRenderVisibleAmount(
        bool negative,
        string integer,
        string cents,
        string currencyStyle,
        string groupedThousands,
        string signPosition,
        out string rendered)
    {
        rendered = string.Empty;
        if (!AmountCurrencyStyles.Contains(currencyStyle, StringComparer.Ordinal)
            || !AmountGroupingStyles.Contains(groupedThousands, StringComparer.Ordinal)
            || !AmountSignPositions.Contains(signPosition, StringComparer.Ordinal)
            || negative != (signPosition != "none")
            || (signPosition == "after_currency" && currencyStyle == "none"))
        {
            return false;
        }
        var visibleInteger = groupedThousands == "grouped_thousands" ? FormatThousands(integer) : integer;
        var numeric = $"{visibleInteger}.{cents}";
        var currency = currencyStyle switch
        {
            "none" => string.Empty,
            "yen" => "¥",
            "yen_space" => "¥ ",
            "fullwidth_yen" => "￥",
            "fullwidth_yen_space" => "￥ ",
            _ => throw new InvalidOperationException("Unsupported verified amount currency style"),
        };
        rendered = signPosition switch
        {
            "before_currency_or_number" => "-" + currency + numeric,
            "after_currency" => currency.TrimEnd() + "-" + (currency.EndsWith(' ') ? " " : string.Empty) + numeric,
            _ => currency + numeric,
        };
        return true;
    }

    private static string FormatThousands(string digits)
    {
        var groups = new List<string>();
        for (var end = digits.Length; end > 0; end -= 3)
        {
            var start = Math.Max(0, end - 3);
            groups.Add(digits[start..end]);
        }
        groups.Reverse();
        return string.Join(',', groups);
    }

    private static bool IsValidTimeDisplay(string candidate)
    {
        if (TimeOnly.TryParseExact(candidate, ["H:mm", "HH:mm", "H:mm:ss", "HH:mm:ss"], System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out _))
        {
            return true;
        }
        return DateTime.TryParseExact(
            candidate,
            ["yyyy-MM-dd HH:mm", "yyyy-MM-dd HH:mm:ss"],
            System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.None,
            out _);
    }

    private static ClassRead DecodeClass(OrtOutput output)
    {
        if (output.Shape.Length != 1)
        {
            throw new InvalidOperationException("Unified OCR finite classifier tensor must be one-dimensional");
        }
        return ArgMax(output.Values, 0, output.Shape[0]);
    }

    private static ClassRead ArgMax(float[] values, int offset, int count)
    {
        if (count <= 0 || offset < 0 || offset + count > values.Length)
        {
            throw new InvalidOperationException("Unified OCR output contains an invalid score vector");
        }
        var maximum = values[offset];
        if (!float.IsFinite(maximum))
        {
            throw new InvalidOperationException("Unified OCR output contains a non-finite score");
        }
        var maximumIndex = 0;
        for (var index = 1; index < count; index++)
        {
            var value = values[offset + index];
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("Unified OCR output contains a non-finite score");
            }
            if (value > maximum)
            {
                maximum = value;
                maximumIndex = index;
            }
        }
        var denominator = 0.0;
        for (var index = 0; index < count; index++)
        {
            denominator += Math.Exp(values[offset + index] - maximum);
        }
        if (!double.IsFinite(denominator) || denominator <= 0.0)
        {
            throw new InvalidOperationException("Unified OCR output has an invalid softmax denominator");
        }
        var confidence = (float)(1.0 / denominator);
        return new ClassRead(maximumIndex, confidence);
    }

    private void VerifyRuntimeAbi()
    {
        var inputNames = _session.InputMetadata.Keys.ToArray();
        if (!HasExactNames(inputNames, InputNames))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime inputs differ from its v12 contract: [{string.Join(',', inputNames)}]");
        }
        VerifyRuntimeShape(_session.InputMetadata[FieldImagesInput].Dimensions, [5, 1, _bundle.FieldHeight, _bundle.FieldWidth], FieldImagesInput);
        VerifyRuntimeShape(_session.InputMetadata[RecipientValueInput].Dimensions, [1, 1, _bundle.RecipientHeight, _bundle.RecipientWidth], RecipientValueInput);

        var outputNames = _session.OutputMetadata.Keys.ToArray();
        if (!HasExactNames(outputNames, UnifiedOcrBundle.OutputNames))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime outputs differ from its v12 contract: [{string.Join(',', outputNames)}]");
        }
        foreach (var outputName in UnifiedOcrBundle.OutputNames)
        {
            VerifyRuntimeShape(_session.OutputMetadata[outputName].Dimensions, _bundle.OutputShapes[outputName], outputName);
        }
    }

    private static void VerifyRuntimeShape(IReadOnlyList<int> actual, IReadOnlyList<int> expected, string name)
    {
        if (!actual.SequenceEqual(expected))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime tensor {name} has shape [{string.Join(',', actual)}], expected [{string.Join(',', expected)}]");
        }
    }

    /// <summary>
    /// ONNX Runtime exposes metadata/output collections, but does not promise
    /// their enumeration order. The v12 ABI is a fixed named ABI, so require
    /// exactly the names from the contract without treating collection order
    /// as part of the runtime contract.
    /// </summary>
    private static bool HasExactNames(IEnumerable<string> actual, IReadOnlyCollection<string> expected)
    {
        var names = actual.ToArray();
        return names.Length == expected.Count
            && names.Distinct(StringComparer.Ordinal).Count() == names.Length
            && names.ToHashSet(StringComparer.Ordinal).SetEquals(expected);
    }

    private static InferenceSession CreateSession(string modelPath, DeviceSetting device, out string provider)
    {
        if (device.GpuDeviceId is null)
        {
            provider = "cpu";
            return new InferenceSession(modelPath);
        }
        try
        {
            using var options = new SessionOptions();
            options.AppendExecutionProvider_CUDA(device.GpuDeviceId.Value);
            var session = new InferenceSession(modelPath, options);
            provider = $"cuda:{device.GpuDeviceId.Value}";
            return session;
        }
        catch when (device.FallbackToCpu)
        {
            provider = "cpu (auto fallback)";
            return new InferenceSession(modelPath);
        }
    }

    private sealed record OrtOutput(float[] Values, int[] Shape);
    private sealed record CtcRead(string Text, float Confidence);
    private sealed record StructuredRead(string Text, float Confidence);
    private sealed record ClassRead(int Index, float Confidence);
}
