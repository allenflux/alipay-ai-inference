using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

internal sealed record WhiteLineStudentRead(string Text, float Confidence);

/// <summary>
/// CPU-only CTC student for one PP-OCR DB/CLS-oriented line crop. The caller
/// owns the crop; this engine reads it without altering or disposing it.
/// </summary>
internal sealed class WhiteLineStudentEngine : IDisposable
{
    private readonly WhiteLineStudentBundle _bundle;
    private readonly InferenceSession _session;
    private bool _disposed;

    public WhiteLineStudentEngine(WhiteLineStudentBundle bundle)
    {
        ArgumentNullException.ThrowIfNull(bundle);
        _bundle = bundle;
        var session = bundle.OpenCpuSession();
        try
        {
            VerifyAbi(session, bundle);
            _session = session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    public string ExecutionProvider => "cpu";

    public WhiteLineStudentRead Recognize(Mat paddleDbClsOrientedRgbCrop)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(WhiteLineStudentEngine));
        }
        ArgumentNullException.ThrowIfNull(paddleDbClsOrientedRgbCrop);
        if (paddleDbClsOrientedRgbCrop.Empty()
            || paddleDbClsOrientedRgbCrop.Type() != MatType.CV_8UC3)
        {
            throw new InvalidOperationException(
                "White line student requires one non-empty PP-OCR DB/CLS-oriented RGB crop");
        }

        var input = PrepareInput(
            paddleDbClsOrientedRgbCrop,
            _bundle.ImageHeight,
            _bundle.ImageWidth);
        var tensor = new DenseTensor<float>(
            input,
            [1, 1, _bundle.ImageHeight, _bundle.ImageWidth]);
        using var outputs = _session.Run(
            [NamedOnnxValue.CreateFromTensor(WhiteLineStudentBundle.InputName, tensor)]);
        var selected = outputs.FirstOrDefault(output =>
            string.Equals(output.Name, WhiteLineStudentBundle.OutputName, StringComparison.Ordinal));
        if (selected is null)
        {
            throw new InvalidOperationException(
                $"White line student did not return {WhiteLineStudentBundle.OutputName}");
        }
        var logits = selected.AsTensor<float>();
        var dimensions = logits.Dimensions.ToArray();
        if (dimensions.Length != 3
            || dimensions[0] <= 0
            || dimensions[1] != 1
            || dimensions[2] != _bundle.Characters.Count + 1)
        {
            throw new InvalidOperationException(
                $"White line student logits must be [time,1,charset+blank], got [{string.Join(',', dimensions)}]");
        }
        var decoded = UnifiedCtcDecoder.Decode(
            logits.ToArray(),
            dimensions[0],
            dimensions[2],
            _bundle.Characters);
        return new WhiteLineStudentRead(decoded.Text, decoded.Confidence);
    }

    internal static float[] PrepareInput(Mat rgb, int targetHeight, int targetWidth)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (rgb.Empty() || rgb.Type() != MatType.CV_8UC3)
        {
            throw new InvalidOperationException("White line student preprocessing requires CV_8UC3 RGB pixels");
        }
        if (targetHeight <= 0 || targetWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(targetHeight),
                "White line student target dimensions must be positive");
        }

        using var grayscale = new Mat(rgb.Rows, rgb.Cols, MatType.CV_8UC1);
        for (var y = 0; y < rgb.Rows; y++)
        {
            for (var x = 0; x < rgb.Cols; x++)
            {
                var pixel = rgb.At<Vec3b>(y, x);
                // Integer ITU-R 601 coefficients are part of
                // opencv_exact_rgb_gray_letterbox_v1; no platform floating
                // point or library colour-conversion rounding is involved.
                grayscale.Set(y, x, RgbToGray(pixel.Item0, pixel.Item1, pixel.Item2));
            }
        }

        var scale = Math.Min(targetWidth / (double)grayscale.Cols, targetHeight / (double)grayscale.Rows);
        var resizedWidth = Math.Clamp(
            (int)Math.Round(grayscale.Cols * scale, MidpointRounding.ToEven),
            1,
            targetWidth);
        var resizedHeight = Math.Clamp(
            (int)Math.Round(grayscale.Rows * scale, MidpointRounding.ToEven),
            1,
            targetHeight);
        using var resized = new Mat();
        Cv2.Resize(
            grayscale,
            resized,
            new OpenCvSharp.Size(resizedWidth, resizedHeight),
            0,
            0,
            InterpolationFlags.LinearExact);

        var values = Enumerable.Repeat(1.0f, checked(targetHeight * targetWidth)).ToArray();
        var left = (targetWidth - resizedWidth) / 2;
        var top = (targetHeight - resizedHeight) / 2;
        for (var y = 0; y < resizedHeight; y++)
        {
            for (var x = 0; x < resizedWidth; x++)
            {
                values[(top + y) * targetWidth + left + x] = resized.At<byte>(y, x) / 255.0f;
            }
        }
        return values;
    }

    internal static byte RgbToGray(byte red, byte green, byte blue)
    {
        var value = checked(
            red * 19595
            + green * 38470
            + blue * 7471
            + 32768) >> 16;
        return (byte)value;
    }

    private static void VerifyAbi(InferenceSession session, WhiteLineStudentBundle bundle)
    {
        if (!session.InputNames.SequenceEqual([WhiteLineStudentBundle.InputName], StringComparer.Ordinal)
            || !session.OutputNames.SequenceEqual([WhiteLineStudentBundle.OutputName], StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                "White line student must expose exactly image -> logits");
        }
        var input = session.InputMetadata[WhiteLineStudentBundle.InputName];
        var expectedInput = new[] { 1, 1, bundle.ImageHeight, bundle.ImageWidth };
        if (!input.IsTensor
            || input.ElementType != typeof(float)
            || !input.Dimensions.SequenceEqual(expectedInput))
        {
            throw new InvalidOperationException(
                $"White line student input must be float [{string.Join(',', expectedInput)}]");
        }
        var output = session.OutputMetadata[WhiteLineStudentBundle.OutputName];
        if (!output.IsTensor
            || output.ElementType != typeof(float)
            || output.Dimensions.Length != 3
            || output.Dimensions[0] <= 0
            || output.Dimensions[1] != 1
            || output.Dimensions[2] != bundle.Characters.Count + 1)
        {
            throw new InvalidOperationException(
                "White line student output must be static float [time,1,charset+blank]");
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _session.Dispose();
    }
}
