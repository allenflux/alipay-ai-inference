using System.Runtime.InteropServices;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Full-image-only rectification that mirrors geometry.py's deterministic
/// portrait orientation plus full_image_quad + warp_quad(max_side=1600) path.
///
/// This deliberately does not try to find a phone/screen boundary.  Production
/// callers provide an EXIF-upright full receipt image.  Landscape inputs are
/// rotated 90 degrees clockwise before the cubic warp, matching the geometry
/// used to train and evaluate the detector.
/// </summary>
internal static class ReceiptRectifier
{
    public const string NoneMode = "none";
    public const string MaxSide1600Mode = "max-side-1600";
    public const int MaximumSide = 1600;

    public static ReceiptRectification Rectify(Image<Rgb24> source, string mode)
    {
        ArgumentNullException.ThrowIfNull(source);
        return mode switch
        {
            NoneMode => ReceiptRectification.Identity(source),
            MaxSide1600Mode => WarpFullImage(source, MaximumSide),
            _ => throw new ArgumentException($"Unsupported rectification mode: {mode}", nameof(mode)),
        };
    }

    private static ReceiptRectification WarpFullImage(Image<Rgb24> source, int maxSide)
    {
        if (source.Width < 2 || source.Height < 2)
        {
            throw new InvalidOperationException(
                "--rectification max-side-1600 requires an EXIF-upright image at least 2x2 pixels");
        }

        var rotationDegrees = source.Width > source.Height ? 90 : 0;
        var rotatedWidth = rotationDegrees == 90 ? source.Height : source.Width;
        var rotatedHeight = rotationDegrees == 90 ? source.Width : source.Height;
        var outputWidth = rotatedWidth;
        var outputHeight = rotatedHeight;
        var longestSide = Math.Max(outputWidth, outputHeight);
        if (longestSide > maxSide)
        {
            var scale = (double)maxSide / longestSide;
            outputWidth = Math.Max(2, (int)Math.Round(outputWidth * scale, MidpointRounding.ToEven));
            outputHeight = Math.Max(2, (int)Math.Round(outputHeight * scale, MidpointRounding.ToEven));
        }

        // Python full_image_quad uses endpoint pixel coordinates, not width and
        // height.  Point2f plus GetPerspectiveTransform therefore reproduces
        // cv2.getPerspectiveTransform(...astype(np.float32), ...astype(np.float32)).
        var sourceQuad = FullImageQuad(rotatedWidth, rotatedHeight);
        var destinationQuad = FullImageQuad(outputWidth, outputHeight);
        using var rotatedToRectifiedMat = Cv2.GetPerspectiveTransform(sourceQuad, destinationQuad);
        var originalToRotated = RotationHomography(source.Height, rotationDegrees);
        var originalToRectified = MultiplyMatrices(ReadMatrix(rotatedToRectifiedMat), originalToRotated);
        var rectifiedToOriginal = InvertMatrix(originalToRectified);

        using var sourceMat = PaddleOcrImageOps.ToRgbMat(source);
        using var rotatedMat = new Mat();
        var warpSource = sourceMat;
        if (rotationDegrees == 90)
        {
            // Keep the right-angle rotation as a discrete pixel operation.
            // Python executes cv2.rotate first and only then applies the cubic
            // warp; combining them would add an interpolation and change model
            // input pixels.
            Cv2.Rotate(sourceMat, rotatedMat, RotateFlags.Rotate90Clockwise);
            warpSource = rotatedMat;
        }
        using var rectifiedMat = new Mat();
        // Execute WarpPerspective even when the image already fits within 1600.
        // That is the observable pixel contract of Python warp_quad.
        Cv2.WarpPerspective(
            warpSource,
            rectifiedMat,
            rotatedToRectifiedMat,
            new OpenCvSharp.Size(outputWidth, outputHeight),
            InterpolationFlags.Cubic,
            BorderTypes.Replicate);

        var rectified = ToRgbImage(rectifiedMat);
        return new ReceiptRectification(
            rectified,
            source.Width,
            source.Height,
            MaxSide1600Mode,
            rotationDegrees,
            originalToRectified,
            rectifiedToOriginal,
            ownsImage: true);
    }

    private static double[,] RotationHomography(int height, int degrees)
    {
        return degrees switch
        {
            0 => new double[,]
            {
                { 1.0, 0.0, 0.0 },
                { 0.0, 1.0, 0.0 },
                { 0.0, 0.0, 1.0 },
            },
            90 => new double[,]
            {
                { 0.0, -1.0, height - 1.0 },
                { 1.0, 0.0, 0.0 },
                { 0.0, 0.0, 1.0 },
            },
            _ => throw new InvalidOperationException($"Unsupported production rotation: {degrees}"),
        };
    }

    private static double[,] MultiplyMatrices(double[,] left, double[,] right)
    {
        var product = new double[3, 3];
        for (var row = 0; row < 3; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                for (var inner = 0; inner < 3; inner++)
                {
                    product[row, column] += left[row, inner] * right[inner, column];
                }
            }
        }
        return product;
    }

    private static double[,] InvertMatrix(double[,] matrix)
    {
        var a = matrix[0, 0];
        var b = matrix[0, 1];
        var c = matrix[0, 2];
        var d = matrix[1, 0];
        var e = matrix[1, 1];
        var f = matrix[1, 2];
        var g = matrix[2, 0];
        var h = matrix[2, 1];
        var i = matrix[2, 2];
        var determinant = a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g);
        if (!double.IsFinite(determinant) || Math.Abs(determinant) < 1e-12)
        {
            throw new InvalidOperationException("Full-image rectification homography is singular");
        }
        var inverseScale = 1.0 / determinant;
        return new double[,]
        {
            { (e * i - f * h) * inverseScale, (c * h - b * i) * inverseScale, (b * f - c * e) * inverseScale },
            { (f * g - d * i) * inverseScale, (a * i - c * g) * inverseScale, (c * d - a * f) * inverseScale },
            { (d * h - e * g) * inverseScale, (b * g - a * h) * inverseScale, (a * e - b * d) * inverseScale },
        };
    }

    private static Point2f[] FullImageQuad(int width, int height)
    {
        return
        [
            new Point2f(0, 0),
            new Point2f(width - 1, 0),
            new Point2f(width - 1, height - 1),
            new Point2f(0, height - 1),
        ];
    }

    private static double[,] ReadMatrix(Mat matrix)
    {
        if (matrix.Rows != 3 || matrix.Cols != 3 || matrix.Type() != MatType.CV_64FC1)
        {
            throw new InvalidOperationException("OpenCV returned an invalid rectification homography");
        }
        var values = new double[3, 3];
        for (var row = 0; row < 3; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                values[row, column] = matrix.At<double>(row, column);
            }
        }
        return values;
    }

    private static Image<Rgb24> ToRgbImage(Mat rgb)
    {
        if (rgb.Empty() || rgb.Type() != MatType.CV_8UC3)
        {
            throw new InvalidOperationException("OpenCV returned an invalid rectified RGB image");
        }
        if (!rgb.IsContinuous())
        {
            throw new InvalidOperationException("OpenCV returned a non-contiguous rectified RGB image");
        }
        var bytes = new byte[checked(rgb.Width * rgb.Height * 3)];
        Marshal.Copy(rgb.Data, bytes, 0, bytes.Length);
        return Image.LoadPixelData<Rgb24>(bytes, rgb.Width, rgb.Height);
    }
}

internal sealed class ReceiptRectification : IDisposable
{
    private readonly double[,] _originalToRectified;
    private readonly double[,] _rectifiedToOriginal;
    private readonly bool _ownsImage;

    public ReceiptRectification(
        Image<Rgb24> image,
        int sourceWidth,
        int sourceHeight,
        string mode,
        int rotationDegrees,
        double[,] originalToRectified,
        double[,] rectifiedToOriginal,
        bool ownsImage)
    {
        Image = image;
        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        Mode = mode;
        RotationDegrees = rotationDegrees;
        _originalToRectified = originalToRectified;
        _rectifiedToOriginal = rectifiedToOriginal;
        _ownsImage = ownsImage;
    }

    public Image<Rgb24> Image { get; }
    public int SourceWidth { get; }
    public int SourceHeight { get; }
    public string Mode { get; }
    public int RotationDegrees { get; }

    public static ReceiptRectification Identity(Image<Rgb24> source)
    {
        var identity = IdentityMatrix();
        return new ReceiptRectification(
            source,
            source.Width,
            source.Height,
            ReceiptRectifier.NoneMode,
            rotationDegrees: 0,
            identity,
            IdentityMatrix(),
            ownsImage: false);
    }

    public float[] ProjectBoxToSource(float[] rectifiedBox)
    {
        if (rectifiedBox.Length < 4 || rectifiedBox.Take(4).Any(value => !float.IsFinite(value)))
        {
            throw new ArgumentException("A rectified box must contain four finite xyxy values", nameof(rectifiedBox));
        }
        var x1 = rectifiedBox[0];
        var y1 = rectifiedBox[1];
        var x2 = rectifiedBox[2];
        var y2 = rectifiedBox[3];
        var corners = new[]
        {
            Transform(x1, y1, _rectifiedToOriginal),
            Transform(x2, y1, _rectifiedToOriginal),
            Transform(x2, y2, _rectifiedToOriginal),
            Transform(x1, y2, _rectifiedToOriginal),
        };
        return
        [
            (float)Math.Clamp(corners.Min(point => point.X), 0.0, SourceWidth),
            (float)Math.Clamp(corners.Min(point => point.Y), 0.0, SourceHeight),
            (float)Math.Clamp(corners.Max(point => point.X), 0.0, SourceWidth),
            (float)Math.Clamp(corners.Max(point => point.Y), 0.0, SourceHeight),
        ];
    }

    public RectificationGeometry Geometry()
    {
        return new RectificationGeometry(
            new ImageSize(SourceWidth, SourceHeight),
            new ImageSize(Image.Width, Image.Height),
            Mode,
            RotationDegrees,
            ScreenDetected: false,
            FullImageQuadForJson(SourceWidth, SourceHeight),
            MatrixForJson(_originalToRectified),
            MatrixForJson(_rectifiedToOriginal));
    }

    public void Dispose()
    {
        if (_ownsImage)
        {
            Image.Dispose();
        }
    }

    private static (double X, double Y) Transform(double x, double y, double[,] matrix)
    {
        var divisor = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2];
        if (!double.IsFinite(divisor) || Math.Abs(divisor) < 1e-12)
        {
            throw new InvalidOperationException("Rectification homography projected a box point to infinity");
        }
        var transformedX = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / divisor;
        var transformedY = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / divisor;
        if (!double.IsFinite(transformedX) || !double.IsFinite(transformedY))
        {
            throw new InvalidOperationException("Rectification homography produced a non-finite box point");
        }
        return (transformedX, transformedY);
    }

    private static double[,] IdentityMatrix()
    {
        return new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
        };
    }

    private static double[][] MatrixForJson(double[,] matrix)
    {
        var output = new double[3][];
        for (var row = 0; row < 3; row++)
        {
            output[row] = new double[3];
            for (var column = 0; column < 3; column++)
            {
                // Match RectificationResult.manifest(), which rounds audit
                // matrices to eight decimal places but keeps full precision
                // internally for coordinate projection.
                output[row][column] = Math.Round(matrix[row, column], 8, MidpointRounding.ToEven);
            }
        }
        return output;
    }

    private static float[][] FullImageQuadForJson(int width, int height)
    {
        return
        [
            [0.0f, 0.0f],
            [width - 1.0f, 0.0f],
            [width - 1.0f, height - 1.0f],
            [0.0f, height - 1.0f],
        ];
    }
}

internal sealed record RectificationGeometry(
    ImageSize SourceSize,
    ImageSize RectifiedSize,
    string Rectification,
    int RotationDegrees,
    bool ScreenDetected,
    float[][] ScreenQuadOriginal,
    [property: System.Text.Json.Serialization.JsonPropertyName("H_original_to_rectified")]
    double[][] HOriginalToRectified,
    [property: System.Text.Json.Serialization.JsonPropertyName("H_rectified_to_original")]
    double[][] HRectifiedToOriginal);
