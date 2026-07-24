using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>One recognised PaddleOCR text line, after CTC decoding.</summary>
internal sealed record PaddleOcrLine(string Text, float Confidence);

/// <summary>Aggregate matching the Python reader's text/confidence semantics.</summary>
internal sealed record PaddleOcrReadResult(string Text, float? Confidence, IReadOnlyList<PaddleOcrLine> Lines);

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
        using var prepared = PaddleOcrImageOps.ResizeKeepRatio(rgb, shape.Height, shape.Width, padRight: true);
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
                    padRight: true);
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
