using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Diagnostic result emitted by a v12/v13 unified OCR reader. The delivery
/// value is deliberately separate from the candidate: current unified text
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
    string? StatusNormalized,
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
/// One-session implementation of architecture-v12/v13 unified receipt OCR.
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
    private readonly float[] _fieldValues;
    private readonly float[] _recipientValues;

    public UnifiedOcrEngine(UnifiedOcrBundle bundle, DeviceSetting requestedDevice)
    {
        _bundle = bundle;
        // ReceiptMlNetProgram owns one engine and invokes it from its serial
        // image loop. Reuse the fixed-shape ABI buffers instead of allocating
        // them for every receipt; this engine must not be shared concurrently.
        _fieldValues = new float[checked(5 * bundle.FieldHeight * bundle.FieldWidth)];
        _recipientValues = new float[checked(bundle.RecipientHeight * bundle.RecipientWidth)];
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
    public int ArchitectureVersion => _bundle.ArchitectureVersion;

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
        Array.Fill(_fieldValues, 1.0f);
        Array.Fill(_recipientValues, 1.0f);

        var readable = new HashSet<string>(StringComparer.Ordinal);
        WriteField(byLabel.GetValueOrDefault("amount"), 0, rightAlign: true, source, _fieldValues, readable, "amount");
        WriteField(byLabel.GetValueOrDefault("time"), 1, rightAlign: true, source, _fieldValues, readable, "time");
        WriteField(byLabel.GetValueOrDefault("transfer_status"), 2, rightAlign: false, source, _fieldValues, readable, "transfer_status");
        WriteField(byLabel.GetValueOrDefault("payment_method_field"), 3, rightAlign: true, source, _fieldValues, readable, "payment_method_field");
        // Architectures v12/v13 freeze channel 4 as white. Recipient text is read
        // only from the separate high-resolution input below.
        WriteRecipient(byLabel.GetValueOrDefault("recipient_field"), source, _recipientValues, readable);

        var fieldTensor = new DenseTensor<float>(_fieldValues, [5, 1, _bundle.FieldHeight, _bundle.FieldWidth]);
        var recipientTensor = new DenseTensor<float>(_recipientValues, [1, 1, _bundle.RecipientHeight, _bundle.RecipientWidth]);
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor(FieldImagesInput, fieldTensor),
            NamedOnnxValue.CreateFromTensor(RecipientValueInput, recipientTensor),
        };
        var preprocessMs = StopAndReadMilliseconds(stageStopwatch);

        UnifiedOcrReadResult result;
        double inferenceMs;
        double postprocessMs;
        stageStopwatch.Restart();
        using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runtimeOutputs = _session.Run(inputs))
        {
            inferenceMs = StopAndReadMilliseconds(stageStopwatch);
            stageStopwatch.Restart();
            // DisposableNamedOnnxValue already exposes ORT's native CPU output
            // memory through DenseTensor.Buffer. Decode while that collection
            // is alive instead of copying every output tensor to a new array.
            var outputs = ReadOutputViews(runtimeOutputs);

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
            string? statusNormalized = null;
            float? statusConfidence = null;
            if (readable.Contains("transfer_status"))
            {
                if (_bundle.StatusTextCharacters is not null)
                {
                    var output = outputs[UnifiedOcrBundle.StatusTextOutputName];
                    var statusText = UnifiedStatusTextDecoder.Decode(
                        output.Values.Span,
                        output.Shape[0],
                        output.Shape[1],
                        _bundle.StatusTextCharacters);
                    if (!string.IsNullOrWhiteSpace(statusText.Text))
                    {
                        statusCandidate = statusText.Text;
                        statusNormalized = statusText.Normalized;
                        statusConfidence = statusText.Confidence;
                    }
                }
                else if (_bundle.StatusRuntimePolicy == "classify")
                {
                    // Legacy v12 classifiers are consumed only when their
                    // audited contract explicitly permits classification.
                    // review_only logits are untrained diagnostics and must
                    // never escape as a status candidate.
                    var status = DecodeClass(outputs["status_logits"]);
                    statusCandidate = _bundle.StatusClasses[status.Index];
                    statusNormalized = statusCandidate;
                    statusConfidence = status.Confidence;
                }
            }

            var effectiveStatusPolicy = _bundle.HasStatusTextCtc
                ? UnifiedOcrBundle.StatusTextRuntimePolicy
                : _bundle.StatusRuntimePolicy;
            var statusDeliveryValue = !_bundle.HasStatusTextCtc && _bundle.StatusRuntimePolicy == "classify"
                ? statusNormalized ?? "review"
                : _bundle.HasStatusTextCtc
                    ? "review"
                    : _bundle.StatusReviewValue ?? "review";

            result = new UnifiedOcrReadResult(
                candidates,
                statusCandidate,
                statusNormalized,
                statusConfidence,
                statusDeliveryValue,
                effectiveStatusPolicy,
                _bundle.TextReviewValue,
                _bundle.TextRuntimePolicy);
        }
        // Keep the historical stage boundary: output disposal is part of OCR
        // postprocessing telemetry even though no tensor view escapes it.
        postprocessMs = StopAndReadMilliseconds(stageStopwatch);

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
        UnifiedOcrImageOps.WriteFieldTensor(
            crop,
            _bundle.FieldHeight,
            _bundle.FieldWidth,
            rightAlign,
            destination,
            checked(slot * _bundle.FieldHeight * _bundle.FieldWidth));
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
        UnifiedOcrImageOps.WriteFieldTensor(
            crop,
            _bundle.RecipientHeight,
            _bundle.RecipientWidth,
            rightAlign: false,
            destination: destination,
            destinationOffset: 0,
            leftCropFraction: _bundle.RecipientLeftTrim);
        readable.Add("recipient_field");
    }

    private Dictionary<string, OrtOutput> ReadOutputViews(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runtimeOutputs)
    {
        var names = runtimeOutputs.Select(output => output.Name).ToArray();
        if (!HasExactNames(names, _bundle.OutputNames))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime outputs differ from its architecture-v{_bundle.ArchitectureVersion} contract: [{string.Join(',', names)}]");
        }

        var output = new Dictionary<string, OrtOutput>(StringComparer.Ordinal);
        foreach (var namedValue in runtimeOutputs)
        {
            var tensor = namedValue.AsTensor<float>();
            if (tensor is not DenseTensor<float> denseTensor)
            {
                throw new InvalidOperationException(
                    $"Unified OCR runtime output {namedValue.Name} is not a dense CPU tensor");
            }
            var shape = tensor.Dimensions.ToArray();
            if (!_bundle.OutputShapes.TryGetValue(namedValue.Name, out var expected)
                || !shape.SequenceEqual(expected))
            {
                throw new InvalidOperationException(
                    $"Unified OCR runtime output {namedValue.Name} has invalid static shape [{string.Join(',', shape)}]");
            }
            output.Add(namedValue.Name, new OrtOutput(denseTensor.Buffer, shape));
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
        var decoded = UnifiedCtcDecoder.Decode(output.Values.Span, output.Shape[0], output.Shape[1], characters);
        return new CtcRead(decoded.Text, decoded.Confidence);
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
            var decoded = ArgMax(digitsOutput.Values.Span, index * 10, 10);
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
        return ArgMax(output.Values.Span, 0, output.Shape[0]);
    }

    private static ClassRead ArgMax(ReadOnlySpan<float> values, int offset, int count)
    {
        var maximumIndex = UnifiedCtcDecoder.FindMaximumIndex(values, offset, count, out var maximum);
        var confidence = UnifiedCtcDecoder.WinningSoftmaxConfidence(values, offset, count, maximum);
        return new ClassRead(maximumIndex, confidence);
    }

    private void VerifyRuntimeAbi()
    {
        var inputNames = _session.InputMetadata.Keys.ToArray();
        if (!HasExactNames(inputNames, InputNames))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime inputs differ from its architecture-v{_bundle.ArchitectureVersion} contract: [{string.Join(',', inputNames)}]");
        }
        VerifyRuntimeShape(_session.InputMetadata[FieldImagesInput].Dimensions, [5, 1, _bundle.FieldHeight, _bundle.FieldWidth], FieldImagesInput);
        VerifyRuntimeShape(_session.InputMetadata[RecipientValueInput].Dimensions, [1, 1, _bundle.RecipientHeight, _bundle.RecipientWidth], RecipientValueInput);

        var outputNames = _session.OutputMetadata.Keys.ToArray();
        if (!HasExactNames(outputNames, _bundle.OutputNames))
        {
            throw new InvalidOperationException(
                $"Unified OCR runtime outputs differ from its architecture-v{_bundle.ArchitectureVersion} contract: [{string.Join(',', outputNames)}]");
        }
        foreach (var outputName in _bundle.OutputNames)
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
    /// their enumeration order. The versioned ABI is a fixed named ABI, so require
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

    private sealed record OrtOutput(Memory<float> Values, int[] Shape);
    private sealed record CtcRead(string Text, float Confidence);
    private sealed record StructuredRead(string Text, float Confidence);
    private sealed record ClassRead(int Index, float Confidence);
}
