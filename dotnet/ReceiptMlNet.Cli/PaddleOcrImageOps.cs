using System.Runtime.InteropServices;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Image operations for the frozen PaddleOCR v2 adapter.
///
/// <para>The OCR delivery contract explicitly says that its source is an RGB
/// ndarray passed directly to PaddleOCR v2.  OpenCV calls a three-channel
/// buffer BGR by convention, but these helpers never use a colour conversion:
/// byte 0 remains the model's R plane, byte 1 the G plane and byte 2 the B
/// plane throughout resize, perspective crop and tensor packing.</para>
/// </summary>
internal static class PaddleOcrImageOps
{
    public static Mat ToRgbMat(Image<Rgb24> image)
    {
        var bytes = new byte[checked(image.Width * image.Height * 3)];
        var offset = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < image.Height; y++)
            {
                var rowBytes = MemoryMarshal.AsBytes(accessor.GetRowSpan(y));
                rowBytes.CopyTo(bytes.AsSpan(offset, rowBytes.Length));
                offset += rowBytes.Length;
            }
        });

        var result = new Mat(image.Height, image.Width, MatType.CV_8UC3);
        Marshal.Copy(bytes, 0, result.Data, bytes.Length);
        return result;
    }

    public static Mat ResizeForDetection(Mat rgb, int limitSideLength, string limitType)
    {
        if (rgb.Empty())
        {
            throw new InvalidOperationException("Cannot run OCR on an empty image crop");
        }
        if (limitSideLength <= 0)
        {
            throw new InvalidOperationException("PaddleOCR det_limit_side_len must be positive");
        }

        // Paddle's DetResizeForTest pads unusually tiny crops before it
        // calculates the resize ratio, but preserves the original source
        // dimensions in shape_list for DB coordinate restoration.
        Mat? padded = null;
        var working = rgb;
        if (rgb.Rows + rgb.Cols < 64)
        {
            padded = new Mat(Math.Max(32, rgb.Rows), Math.Max(32, rgb.Cols), MatType.CV_8UC3, Scalar.All(0));
            using (var destination = new Mat(padded, new Rect(0, 0, rgb.Cols, rgb.Rows)))
            {
                rgb.CopyTo(destination);
            }
            working = padded;
        }

        var sourceHeight = working.Rows;
        var sourceWidth = working.Cols;
        var ratio = 1.0;
        var longest = Math.Max(sourceHeight, sourceWidth);
        var shortest = Math.Min(sourceHeight, sourceWidth);
        switch (limitType.ToLowerInvariant())
        {
            case "max":
                if (longest > limitSideLength)
                {
                    ratio = (double)limitSideLength / longest;
                }
                break;
            case "min":
                if (shortest < limitSideLength)
                {
                    ratio = (double)limitSideLength / shortest;
                }
                break;
            case "resize_long":
                ratio = (double)limitSideLength / longest;
                break;
            default:
                throw new InvalidOperationException($"Unsupported PaddleOCR det_limit_type: {limitType}");
        }

        // Paddle's DetResizeForTest uses int() then Python's bankers round to
        // make each spatial axis a multiple of 32.
        var resizedHeight = AlignDetectionAxis((int)(sourceHeight * ratio));
        var resizedWidth = AlignDetectionAxis((int)(sourceWidth * ratio));
        try
        {
            var resized = new Mat();
            Cv2.Resize(working, resized, new OpenCvSharp.Size(resizedWidth, resizedHeight), 0, 0, InterpolationFlags.Linear);
            return resized;
        }
        finally
        {
            padded?.Dispose();
        }
    }

    public static float[] ToNormalizedNchw(Mat rgb, float[] mean, float[] scale, int targetWidth, int targetHeight)
    {
        if (rgb.Type() != MatType.CV_8UC3)
        {
            throw new InvalidOperationException($"Expected RGB CV_8UC3 image, got {rgb.Type()}");
        }
        if (mean.Length != 3 || scale.Length != 3)
        {
            throw new ArgumentException("Paddle OCR normalisation requires exactly three channel values");
        }
        if (targetWidth < rgb.Cols || targetHeight < rgb.Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Padded tensor cannot be smaller than its source image");
        }

        var plane = checked(targetWidth * targetHeight);
        var values = new float[checked(plane * 3)];
        for (var y = 0; y < rgb.Rows; y++)
        {
            for (var x = 0; x < rgb.Cols; x++)
            {
                var pixel = rgb.At<Vec3b>(y, x);
                var destination = y * targetWidth + x;
                values[destination] = (pixel.Item0 / 255.0f - mean[0]) / scale[0];
                values[plane + destination] = (pixel.Item1 / 255.0f - mean[1]) / scale[1];
                values[2 * plane + destination] = (pixel.Item2 / 255.0f - mean[2]) / scale[2];
            }
        }
        return values;
    }

    public static Mat RotateCrop(Mat rgb, Point2f[] points)
    {
        if (points.Length != 4)
        {
            throw new ArgumentException("A Paddle text box must have four points", nameof(points));
        }

        var cropWidth = Math.Max(1, (int)Math.Max(Distance(points[0], points[1]), Distance(points[2], points[3])));
        var cropHeight = Math.Max(1, (int)Math.Max(Distance(points[0], points[3]), Distance(points[1], points[2])));
        var destination = new[]
        {
            new Point2f(0, 0),
            new Point2f(cropWidth, 0),
            new Point2f(cropWidth, cropHeight),
            new Point2f(0, cropHeight),
        };
        using var transform = Cv2.GetPerspectiveTransform(points, destination);
        var crop = new Mat();
        Cv2.WarpPerspective(rgb, crop, transform, new OpenCvSharp.Size(cropWidth, cropHeight), InterpolationFlags.Cubic, BorderTypes.Replicate);
        if (crop.Cols > 0 && crop.Rows / (float)crop.Cols >= 1.5f)
        {
            var rotated = new Mat();
            Cv2.Rotate(crop, rotated, RotateFlags.Rotate90Counterclockwise);
            crop.Dispose();
            return rotated;
        }
        return crop;
    }

    public static Mat ResizeKeepRatio(Mat rgb, int targetHeight, int targetWidth, bool padRight)
    {
        if (targetHeight <= 0 || targetWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetHeight));
        }
        var ratio = rgb.Cols / (float)Math.Max(1, rgb.Rows);
        var resizedWidth = Math.Min(targetWidth, Math.Max(1, (int)Math.Ceiling(targetHeight * ratio)));
        using var resized = new Mat();
        Cv2.Resize(rgb, resized, new OpenCvSharp.Size(resizedWidth, targetHeight), 0, 0, InterpolationFlags.Linear);
        if (!padRight || resizedWidth == targetWidth)
        {
            return resized.Clone();
        }

        var padded = new Mat(targetHeight, targetWidth, MatType.CV_8UC3, Scalar.All(0));
        using var destination = new Mat(padded, new Rect(0, 0, resizedWidth, targetHeight));
        resized.CopyTo(destination);
        return padded;
    }

    private static int AlignDetectionAxis(int value)
    {
        return Math.Max((int)Math.Round(value / 32.0, MidpointRounding.ToEven) * 32, 32);
    }

    private static float Distance(Point2f first, Point2f second)
    {
        var dx = first.X - second.X;
        var dy = first.Y - second.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }
}
