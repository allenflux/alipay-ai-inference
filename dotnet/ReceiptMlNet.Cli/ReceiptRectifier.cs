using System.Runtime.InteropServices;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Full-image-only rectification that mirrors geometry.py's
/// full_image_quad + warp_quad(max_side=1600) path.
///
/// This deliberately does not try to find a phone/screen boundary.  Production
/// callers must provide an already upright full receipt image; EXIF orientation
/// is applied by <see cref="ImagePipeline.LoadUprightRgb"/> before this class is
/// called.
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

        var outputWidth = source.Width;
        var outputHeight = source.Height;
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
        var sourceQuad = FullImageQuad(source.Width, source.Height);
        var destinationQuad = FullImageQuad(outputWidth, outputHeight);
        using var originalToRectifiedMat = Cv2.GetPerspectiveTransform(sourceQuad, destinationQuad);
        using var rectifiedToOriginalMat = new Mat();
        var inverseStatus = Cv2.Invert(originalToRectifiedMat, rectifiedToOriginalMat, DecompTypes.LU);
        if (inverseStatus == 0)
        {
            throw new InvalidOperationException("Full-image rectification homography is singular");
        }

        using var sourceMat = PaddleOcrImageOps.ToRgbMat(source);
        using var rectifiedMat = new Mat();
        // Execute WarpPerspective even when the image already fits within 1600.
        // That is the observable pixel contract of Python warp_quad.
        Cv2.WarpPerspective(
            sourceMat,
            rectifiedMat,
            originalToRectifiedMat,
            new OpenCvSharp.Size(outputWidth, outputHeight),
            InterpolationFlags.Cubic,
            BorderTypes.Replicate);

        var rectified = ToRgbImage(rectifiedMat);
        return new ReceiptRectification(
            rectified,
            source.Width,
            source.Height,
            MaxSide1600Mode,
            ReadMatrix(originalToRectifiedMat),
            ReadMatrix(rectifiedToOriginalMat),
            ownsImage: true);
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
        double[,] originalToRectified,
        double[,] rectifiedToOriginal,
        bool ownsImage)
    {
        Image = image;
        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        Mode = mode;
        _originalToRectified = originalToRectified;
        _rectifiedToOriginal = rectifiedToOriginal;
        _ownsImage = ownsImage;
    }

    public Image<Rgb24> Image { get; }
    public int SourceWidth { get; }
    public int SourceHeight { get; }
    public string Mode { get; }

    public static ReceiptRectification Identity(Image<Rgb24> source)
    {
        var identity = IdentityMatrix();
        return new ReceiptRectification(
            source,
            source.Width,
            source.Height,
            ReceiptRectifier.NoneMode,
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
            RotationDegrees: 0,
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
