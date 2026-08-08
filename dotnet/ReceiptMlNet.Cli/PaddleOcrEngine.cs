using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>One recognised PaddleOCR text line, after CTC decoding.</summary>
internal sealed record PaddleOcrLine(string Text, float Confidence);

/// <summary>Aggregate matching the Python reader's text/confidence semantics.</summary>
internal sealed record PaddleOcrReadResult(string Text, float? Confidence, IReadOnlyList<PaddleOcrLine> Lines);

/// <summary>One immutable point from a diagnostic PaddleOCR text quadrilateral.</summary>
internal sealed record PaddleOcrLayoutPoint(float X, float Y);

/// <summary>
/// One decoded DB text box for diagnostic layout inspection. The quadrilateral
/// remains in the coordinate system of the image passed to the engine and is
/// ordered top-left, top-right, bottom-right, bottom-left.
/// </summary>
internal sealed record PaddleOcrLayoutLine(
    string Text,
    float Confidence,
    IReadOnlyList<PaddleOcrLayoutPoint> Quad,
    bool PassesDropScore);

/// <summary>
/// Raw diagnostic layout plus the exact accepted-line projection used by the
/// legacy aggregate reader. This type is not consumed by production fields.
/// </summary>
internal sealed record PaddleOcrLayoutReadResult(
    string Text,
    float? Confidence,
    IReadOnlyList<PaddleOcrLine> AcceptedLines,
    IReadOnlyList<PaddleOcrLayoutLine> Lines);

/// <summary>
/// Direct ONNX Runtime implementation of the frozen PaddleOCR v2 pipeline:
/// DB text detection, perspective crop, angle classification and CTC text
/// recognition.  It owns exactly three sessions for the lifetime of a batch.
/// </summary>
internal sealed class PaddleOcrEngine : IDisposable
{
    private static readonly float[] DetMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] DetStd = [0.229f, 0.224f, 0.225f];
    private static readonly float[] CenterMean = [0.5f, 0.5f, 0.5f];
    private static readonly float[] CenterStd = [0.5f, 0.5f, 0.5f];

    private readonly PaddleOcrDeliveryBundle _bundle;
    private readonly InferenceSession _detector;
    private readonly InferenceSession _classifier;
    private readonly InferenceSession _recognizer;

    public PaddleOcrEngine(PaddleOcrDeliveryBundle bundle, DeviceSetting requestedDevice)
    {
        _bundle = bundle;
        var sessions = CreateSessions(bundle, requestedDevice, out var provider);
        _detector = sessions.Detector;
        _classifier = sessions.Classifier;
        _recognizer = sessions.Recognizer;
        ExecutionProvider = provider;
        try
        {
            VerifySessionContract(_detector, bundle.DetModel);
            VerifySessionContract(_classifier, bundle.ClsModel);
            VerifySessionContract(_recognizer, bundle.RecModel);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The provider successfully used for all three OCR sessions.</summary>
    public string ExecutionProvider { get; }

    public PaddleOcrReadResult Recognize(Image<Rgb24> image)
    {
        using var rgb = PaddleOcrImageOps.ToRgbMat(image);
        return Recognize(rgb);
    }

    public PaddleOcrReadResult Recognize(Mat rgb)
    {
        if (rgb.Empty())
        {
            return new PaddleOcrReadResult(string.Empty, null, Array.Empty<PaddleOcrLine>());
        }

        var boxes = DetectTextBoxes(rgb);
        if (boxes.Count == 0)
        {
            return new PaddleOcrReadResult(string.Empty, null, Array.Empty<PaddleOcrLine>());
        }

        var orientedCrops = new List<Mat>(boxes.Count);
        try
        {
            foreach (var box in boxes)
            {
                using var textCrop = PaddleOcrImageOps.RotateCrop(rgb, box);
                orientedCrops.Add(ClassifyAngle(textCrop));
            }

            // PaddleOCR's CTC decoder can yield an empty text line with a
            // valid score. PaddleOCRReader excludes its text from the joined
            // string, but keeps it in the arithmetic confidence denominator.
            var acceptedLines = new List<PaddleOcrLine>(boxes.Count);
            foreach (var line in RecognizeLines(orientedCrops))
            {
                if (line is not null && line.Confidence >= _bundle.Settings.DropScore)
                {
                    acceptedLines.Add(line);
                }
            }

            if (acceptedLines.Count == 0)
            {
                return new PaddleOcrReadResult(string.Empty, null, Array.Empty<PaddleOcrLine>());
            }
            return new PaddleOcrReadResult(
                string.Join(" ", acceptedLines
                    .Select(line => ReceiptFieldNormalizer.CleanText(line.Text))
                    .Where(text => text.Length > 0)),
                acceptedLines.Average(line => line.Confidence),
                acceptedLines);
        }
        finally
        {
            foreach (var crop in orientedCrops)
            {
                crop.Dispose();
            }
        }
    }

    /// <summary>
    /// Run the same frozen DB/CLS/REC pipeline while retaining every DB text
    /// quadrilateral. This additive diagnostic API deliberately does not
    /// replace or feed <see cref="Recognize(Mat)"/>.
    /// </summary>
    public PaddleOcrLayoutReadResult RecognizeLayoutDiagnostic(Image<Rgb24> image)
    {
        using var rgb = PaddleOcrImageOps.ToRgbMat(image);
        return RecognizeLayoutDiagnostic(rgb);
    }

    /// <summary>
    /// Mat overload for diagnostic tools that already own an RGB OpenCV image.
    /// </summary>
    public PaddleOcrLayoutReadResult RecognizeLayoutDiagnostic(Mat rgb)
    {
        if (rgb.Empty())
        {
            return AssembleLayoutDiagnostic(
                Array.Empty<Point2f[]>(),
                Array.Empty<PaddleOcrLine?>(),
                _bundle.Settings.DropScore);
        }

        var boxes = DetectTextBoxes(rgb);
        if (boxes.Count == 0)
        {
            return AssembleLayoutDiagnostic(
                boxes,
                Array.Empty<PaddleOcrLine?>(),
                _bundle.Settings.DropScore);
        }

        var orientedCrops = new List<Mat>(boxes.Count);
        try
        {
            foreach (var box in boxes)
            {
                using var textCrop = PaddleOcrImageOps.RotateCrop(rgb, box);
                orientedCrops.Add(ClassifyAngle(textCrop));
            }
            return AssembleLayoutDiagnostic(
                boxes,
                RecognizeLines(orientedCrops),
                _bundle.Settings.DropScore);
        }
        finally
        {
            foreach (var crop in orientedCrops)
            {
                crop.Dispose();
            }
        }
    }

    /// <summary>
    /// Pure layout/read assembly kept internal for deterministic contract
    /// tests. Recognition results are already restored to DB-box order by
    /// <see cref="RecognizeLines(IReadOnlyList{Mat})"/>.
    /// </summary>
    internal static PaddleOcrLayoutReadResult AssembleLayoutDiagnostic(
        IReadOnlyList<Point2f[]> boxes,
        IReadOnlyList<PaddleOcrLine?> recognizedLines,
        float dropScore)
    {
        ArgumentNullException.ThrowIfNull(boxes);
        ArgumentNullException.ThrowIfNull(recognizedLines);
        if (!float.IsFinite(dropScore))
        {
            throw new ArgumentOutOfRangeException(nameof(dropScore), "Paddle OCR drop score must be finite");
        }
        if (boxes.Count != recognizedLines.Count)
        {
            throw new InvalidOperationException(
                $"Paddle OCR diagnostic box/line count differs: boxes={boxes.Count} lines={recognizedLines.Count}");
        }

        var layoutLines = new List<PaddleOcrLayoutLine>(boxes.Count);
        var acceptedLines = new List<PaddleOcrLine>(boxes.Count);
        for (var index = 0; index < boxes.Count; index++)
        {
            var box = boxes[index];
            if (box is null || box.Length != 4 || box.Any(point => !float.IsFinite(point.X) || !float.IsFinite(point.Y)))
            {
                throw new InvalidOperationException($"Paddle OCR diagnostic box {index} is not a finite quadrilateral");
            }
            var line = recognizedLines[index]
                ?? throw new InvalidOperationException($"Paddle OCR diagnostic line {index} was not decoded");
            if (!float.IsFinite(line.Confidence))
            {
                throw new InvalidOperationException($"Paddle OCR diagnostic line {index} has non-finite confidence");
            }

            var passesDropScore = line.Confidence >= dropScore;
            var quad = Array.AsReadOnly(box
                .Select(point => new PaddleOcrLayoutPoint(point.X, point.Y))
                .ToArray());
            layoutLines.Add(new PaddleOcrLayoutLine(
                line.Text,
                line.Confidence,
                quad,
                passesDropScore));
            if (passesDropScore)
            {
                acceptedLines.Add(line);
            }
        }

        if (acceptedLines.Count == 0)
        {
            return new PaddleOcrLayoutReadResult(
                string.Empty,
                null,
                Array.Empty<PaddleOcrLine>(),
                layoutLines.AsReadOnly());
        }
        return new PaddleOcrLayoutReadResult(
            string.Join(" ", acceptedLines
                .Select(line => ReceiptFieldNormalizer.CleanText(line.Text))
                .Where(text => text.Length > 0)),
            acceptedLines.Average(line => line.Confidence),
            acceptedLines.AsReadOnly(),
            layoutLines.AsReadOnly());
    }

    public void Dispose()
    {
        _recognizer.Dispose();
        _classifier.Dispose();
        _detector.Dispose();
    }

    private IReadOnlyList<Point2f[]> DetectTextBoxes(Mat rgb)
    {
        using var resized = PaddleOcrImageOps.ResizeForDetection(
            rgb,
            _bundle.Settings.DetLimitSideLength,
            _bundle.Settings.DetLimitType);
        var tensor = PaddleOcrImageOps.ToNormalizedNchw(
            resized,
            DetMean,
            DetStd,
            resized.Cols,
            resized.Rows);
        var output = Run(_detector, _bundle.DetModel, tensor, [1, 3, resized.Rows, resized.Cols]);
        if (output.Shape.Length != 4 || output.Shape[0] != 1 || output.Shape[1] != 1)
        {
            throw new InvalidOperationException(
                $"Paddle OCR detector output must be [1,1,H,W], got [{string.Join(',', output.Shape)}]");
        }
        var mapHeight = output.Shape[2];
        var mapWidth = output.Shape[3];
        return PaddleDbPostProcessor.Process(
            output.Values,
            mapHeight,
            mapWidth,
            rgb.Rows,
            rgb.Cols,
            new PaddleDbOptions(
                _bundle.Settings.DetDbThreshold,
                _bundle.Settings.DetDbBoxThreshold,
                _bundle.Settings.DetDbUnclipRatio,
                _bundle.Settings.UseDilation,
                _bundle.Settings.DetDbScoreMode));
    }

    private Mat ClassifyAngle(Mat rgb)
    {
        var shape = _bundle.Settings.ClsImageShape;
        // Paddle v2 normalizes only the resized pixels into a zero-initialized
        // float tensor. Padding is therefore normalized-space 0, not a black
        // uint8 pixel which would normalize to -1.
        using var prepared = PaddleOcrImageOps.ResizeKeepRatio(rgb, shape.Height, shape.Width, padRight: false);
        var tensor = PaddleOcrImageOps.ToNormalizedNchw(prepared, CenterMean, CenterStd, shape.Width, shape.Height);
        var output = Run(_classifier, _bundle.ClsModel, tensor, [1, shape.Channels, shape.Height, shape.Width]);
        if (output.Shape.Length < 2 || output.Shape[^1] < 2)
        {
            throw new InvalidOperationException(
                $"Paddle OCR classifier output must contain two angle scores, got [{string.Join(',', output.Shape)}]");
        }
        var classCount = output.Shape[^1];
        var (index, score) = ArgMax(output.Values, 0, classCount);
        if (index == 1 && score > _bundle.Settings.ClsThreshold)
        {
            var rotated = new Mat();
            Cv2.Rotate(rgb, rotated, RotateFlags.Rotate180);
            return rotated;
        }
        return rgb.Clone();
    }

    /// <summary>
    /// Mirrors PaddleOCR's recognizer batching: it sorts text crops by aspect
    /// ratio, chooses one dynamic width per batch, then restores results to
    /// text-box order.  If the exported ONNX fixed its batch axis, each crop
    /// remains a batch of one rather than making an invalid input tensor.
    /// </summary>
    private IReadOnlyList<PaddleOcrLine?> RecognizeLines(IReadOnlyList<Mat> images)
    {
        if (images.Count == 0)
        {
            return Array.Empty<PaddleOcrLine?>();
        }

        var shape = _bundle.Settings.RecImageShape;
        var ordered = images
            .Select((image, index) => new RecognitionCrop(index, image, image.Cols / (float)Math.Max(1, image.Rows)))
            .OrderBy(item => item.AspectRatio)
            .ThenBy(item => item.OriginalIndex)
            .ToArray();
        var lines = new PaddleOcrLine?[images.Count];
        var batchSize = _bundle.RecModel.Input.Shape[0].IsDynamic
            ? _bundle.Settings.RecBatchSize
            : 1;
        batchSize = Math.Max(1, batchSize);

        for (var start = 0; start < ordered.Length; start += batchSize)
        {
            var count = Math.Min(batchSize, ordered.Length - start);
            var batch = ordered.AsSpan(start, count);
            var maximumRatio = shape.Width / (float)shape.Height;
            foreach (var crop in batch)
            {
                maximumRatio = Math.Max(maximumRatio, crop.AspectRatio);
            }

            var inputWidth = _bundle.RecModel.Input.Shape[3].StaticValue
                ?? Math.Max(1, (int)(shape.Height * maximumRatio));
            var valuesPerImage = checked(shape.Channels * shape.Height * inputWidth);
            var input = new float[checked(valuesPerImage * count)];
            for (var index = 0; index < count; index++)
            {
                using var prepared = PaddleOcrImageOps.ResizeKeepRatio(
                    batch[index].Image,
                    shape.Height,
                    inputWidth,
                    padRight: false);
                var tensor = PaddleOcrImageOps.ToNormalizedNchw(
                    prepared,
                    CenterMean,
                    CenterStd,
                    inputWidth,
                    shape.Height);
                tensor.CopyTo(input, index * valuesPerImage);
            }

            var output = Run(
                _recognizer,
                _bundle.RecModel,
                input,
                [count, shape.Channels, shape.Height, inputWidth]);
            if (output.Shape.Length != 3 || output.Shape[0] != count)
            {
                throw new InvalidOperationException(
                    $"Paddle OCR recognizer output must be [{count},T,C], got [{string.Join(',', output.Shape)}]");
            }

            var timeSteps = output.Shape[1];
            var classCount = output.Shape[2];
            if (classCount != _bundle.CtcCharacters.Count)
            {
                throw new InvalidOperationException(
                    $"Paddle OCR recognizer output character count {classCount} differs from contract dictionary {_bundle.CtcCharacters.Count}");
            }
            var valuesPerResult = checked(timeSteps * classCount);
            for (var index = 0; index < count; index++)
            {
                lines[batch[index].OriginalIndex] = DecodeCtcLine(output.Values, index * valuesPerResult, timeSteps, classCount);
            }
        }
        return lines;
    }

    private PaddleOcrLine DecodeCtcLine(float[] values, int offset, int timeSteps, int classCount)
    {
        var characters = new List<string>();
        var confidences = new List<float>();
        var previousIndex = -1;
        for (var time = 0; time < timeSteps; time++)
        {
            var (index, confidence) = ArgMax(values, offset + time * classCount, classCount);
            if (index != 0 && index != previousIndex)
            {
                characters.Add(_bundle.CtcCharacters[index]);
                confidences.Add(confidence);
            }
            previousIndex = index;
        }
        return new PaddleOcrLine(string.Concat(characters), confidences.Count == 0 ? 0.0f : confidences.Average());
    }

    private static OrtOutput Run(InferenceSession session, PaddleOcrModelInfo model, float[] values, int[] dimensions)
    {
        var tensor = new DenseTensor<float>(values, dimensions);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(model.InputName, tensor) };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.Run(inputs);
        var selected = outputs.FirstOrDefault(output => string.Equals(output.Name, model.OutputName, StringComparison.Ordinal));
        if (selected is null)
        {
            throw new InvalidOperationException($"Paddle OCR {model.Role} ONNX did not return contract output {model.OutputName}");
        }
        var outputTensor = selected.AsTensor<float>();
        return new OrtOutput(outputTensor.ToArray(), outputTensor.Dimensions.ToArray());
    }

    private static (int Index, float Score) ArgMax(float[] values, int offset, int count)
    {
        if (count <= 0 || offset < 0 || offset + count > values.Length)
        {
            throw new InvalidOperationException("Paddle OCR ONNX output has an invalid score vector");
        }
        var index = 0;
        var maximum = values[offset];
        for (var item = 1; item < count; item++)
        {
            var candidate = values[offset + item];
            if (candidate > maximum)
            {
                maximum = candidate;
                index = item;
            }
        }
        return (index, maximum);
    }

    private static (InferenceSession Detector, InferenceSession Classifier, InferenceSession Recognizer) CreateSessions(
        PaddleOcrDeliveryBundle bundle,
        DeviceSetting device,
        out string provider)
    {
        if (device.GpuDeviceId is null)
        {
            provider = "cpu";
            return CreateCpuSessions(bundle);
        }

        try
        {
            var sessions = CreateCudaSessions(bundle, device.GpuDeviceId.Value);
            provider = $"cuda:{device.GpuDeviceId.Value}";
            return sessions;
        }
        catch when (device.FallbackToCpu)
        {
            provider = "cpu (auto fallback)";
            return CreateCpuSessions(bundle);
        }
    }

    private static void VerifySessionContract(InferenceSession session, PaddleOcrModelInfo model)
    {
        var inputNames = session.InputMetadata.Keys.ToArray();
        if (inputNames.Length != 1 || !string.Equals(inputNames[0], model.Input.Name, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Paddle OCR {model.Role} runtime inputs differ from its verified contract: [{string.Join(',', inputNames)}]");
        }
        VerifyTensorContract(session.InputMetadata[model.Input.Name], model.Input, model.Role);

        var expectedOutputs = model.Outputs.Select(output => output.Name).ToHashSet(StringComparer.Ordinal);
        var outputNames = session.OutputMetadata.Keys.ToArray();
        if (outputNames.Length != expectedOutputs.Count
            || outputNames.Distinct(StringComparer.Ordinal).Count() != outputNames.Length
            || !outputNames.ToHashSet(StringComparer.Ordinal).SetEquals(expectedOutputs))
        {
            throw new InvalidOperationException(
                $"Paddle OCR {model.Role} runtime outputs differ from its verified contract: [{string.Join(',', outputNames)}]");
        }
        foreach (var output in model.Outputs)
        {
            VerifyTensorContract(session.OutputMetadata[output.Name], output, model.Role);
        }
    }

    private static void VerifyTensorContract(
        NodeMetadata metadata,
        PaddleOcrTensorContract contract,
        string role)
    {
        if (!metadata.IsTensor || metadata.ElementType != typeof(float))
        {
            throw new InvalidOperationException(
                $"Paddle OCR {role} runtime tensor {contract.Name} must be tensor(float)");
        }
        if (metadata.Dimensions.Length != contract.Shape.Count)
        {
            throw new InvalidOperationException(
                $"Paddle OCR {role} runtime tensor {contract.Name} rank differs from its verified contract");
        }
        for (var axis = 0; axis < contract.Shape.Count; axis++)
        {
            var actual = metadata.Dimensions[axis];
            var expected = contract.Shape[axis];
            if ((expected.StaticValue is { } fixedValue && actual != fixedValue)
                || (expected.IsDynamic && actual > 0))
            {
                throw new InvalidOperationException(
                    $"Paddle OCR {role} runtime tensor {contract.Name} axis {axis} differs from its verified contract");
            }
        }
    }

    private static (InferenceSession Detector, InferenceSession Classifier, InferenceSession Recognizer) CreateCpuSessions(
        PaddleOcrDeliveryBundle bundle)
    {
        InferenceSession? detector = null;
        InferenceSession? classifier = null;
        try
        {
            detector = new InferenceSession(bundle.DetModel.FullPath);
            classifier = new InferenceSession(bundle.ClsModel.FullPath);
            var recognizer = new InferenceSession(bundle.RecModel.FullPath);
            return (detector, classifier, recognizer);
        }
        catch
        {
            classifier?.Dispose();
            detector?.Dispose();
            throw;
        }
    }

    private static (InferenceSession Detector, InferenceSession Classifier, InferenceSession Recognizer) CreateCudaSessions(
        PaddleOcrDeliveryBundle bundle,
        int gpuDeviceId)
    {
        InferenceSession? detector = null;
        InferenceSession? classifier = null;
        try
        {
            detector = CreateCudaSession(bundle.DetModel.FullPath, gpuDeviceId);
            classifier = CreateCudaSession(bundle.ClsModel.FullPath, gpuDeviceId);
            var recognizer = CreateCudaSession(bundle.RecModel.FullPath, gpuDeviceId);
            return (detector, classifier, recognizer);
        }
        catch
        {
            classifier?.Dispose();
            detector?.Dispose();
            throw;
        }
    }

    private static InferenceSession CreateCudaSession(string modelPath, int gpuDeviceId)
    {
        using var options = new SessionOptions();
        options.AppendExecutionProvider_CUDA(gpuDeviceId);
        return new InferenceSession(modelPath, options);
    }

    private sealed record OrtOutput(float[] Values, int[] Shape);
    private sealed record RecognitionCrop(int OriginalIndex, Mat Image, float AspectRatio);
}
