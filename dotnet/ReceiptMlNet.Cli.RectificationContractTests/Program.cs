using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

internal static class Program
{
    private static int Main()
    {
        try
        {
            VerifyNoneModeDoesNotCopyOrWarp();
            VerifyRgbMatByteOrder();
            VerifySmallImageIdentityWarp();
            VerifyToEvenMaxSideRounding();
            VerifyPortraitMaxSideAndBackProjection();
            VerifyLandscapeMaxSideAndBackProjection();
            Console.WriteLine("PASS: full-image rectification dimensions, homographies, pixels, and box back-projection match the Python contract.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void VerifyRgbMatByteOrder()
    {
        using var source = new Image<Rgb24>(2, 2);
        source[0, 0] = new Rgb24(1, 2, 3);
        source[1, 0] = new Rgb24(4, 5, 6);
        source[0, 1] = new Rgb24(7, 8, 9);
        source[1, 1] = new Rgb24(10, 11, 12);
        using var mat = PaddleOcrImageOps.ToRgbMat(source);
        Assert(mat.IsContinuous(), "ToRgbMat must return a contiguous Mat");
        var actual = new byte[12];
        Marshal.Copy(mat.Data, actual, 0, actual.Length);
        AssertSequenceEqual(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            actual,
            "ToRgbMat RGB row-major byte order");
    }

    private static void VerifyNoneModeDoesNotCopyOrWarp()
    {
        using var source = CreatePattern(13, 7);
        using var result = ReceiptRectifier.Rectify(source, ReceiptRectifier.NoneMode);
        Assert(ReferenceEquals(source, result.Image), "none mode must preserve the loaded EXIF-upright image object");
        AssertEqual(13, result.Image.Width, "none width");
        AssertEqual(7, result.Image.Height, "none height");
        AssertIdentity(result.Geometry().HOriginalToRectified, "none H_original_to_rectified");
        AssertIdentity(result.Geometry().HRectifiedToOriginal, "none H_rectified_to_original");
        AssertBoxClose([1.25f, 2.5f, 11.0f, 6.0f], result.ProjectBoxToSource([1.25f, 2.5f, 11.0f, 6.0f]), 0.0f, "none box");
    }

    private static void VerifySmallImageIdentityWarp()
    {
        using var source = CreatePattern(11, 7);
        var expectedPixels = PixelBytes(source);
        using var result = ReceiptRectifier.Rectify(source, ReceiptRectifier.MaxSide1600Mode);
        Assert(!ReferenceEquals(source, result.Image), "max-side-1600 must execute WarpPerspective even below the limit");
        AssertEqual(11, result.Image.Width, "small warp width");
        AssertEqual(7, result.Image.Height, "small warp height");
        AssertSequenceEqual(expectedPixels, PixelBytes(result.Image), "small identity-warp pixels");
        var geometry = result.Geometry();
        AssertIdentity(geometry.HOriginalToRectified, "small H_original_to_rectified");
        AssertIdentity(geometry.HRectifiedToOriginal, "small H_rectified_to_original");
        AssertEqual(ReceiptRectifier.MaxSide1600Mode, geometry.Rectification, "small mode");
        Assert(!geometry.ScreenDetected, "full-image mode must not claim screen detection");
    }

    private static void VerifyPortraitMaxSideAndBackProjection()
    {
        VerifyScaledCase(sourceWidth: 1179, sourceHeight: 2556, expectedWidth: 738, expectedHeight: 1600);
    }

    private static void VerifyToEvenMaxSideRounding()
    {
        // 5 * 1600 / 3200 = 2.5 -> 2, while 7 * 1600 / 3200 = 3.5 -> 4.
        // These two cases pin Python/.NET midpoint-to-even parity.
        VerifyScaledCase(sourceWidth: 5, sourceHeight: 3200, expectedWidth: 2, expectedHeight: 1600);
        VerifyScaledCase(sourceWidth: 7, sourceHeight: 3200, expectedWidth: 4, expectedHeight: 1600);
    }

    private static void VerifyLandscapeMaxSideAndBackProjection()
    {
        VerifyScaledCase(sourceWidth: 2556, sourceHeight: 1179, expectedWidth: 1600, expectedHeight: 738);
    }

    private static void VerifyScaledCase(int sourceWidth, int sourceHeight, int expectedWidth, int expectedHeight)
    {
        using var source = CreatePattern(sourceWidth, sourceHeight);
        using var result = ReceiptRectifier.Rectify(source, ReceiptRectifier.MaxSide1600Mode);
        AssertEqual(expectedWidth, result.Image.Width, "scaled width");
        AssertEqual(expectedHeight, result.Image.Height, "scaled height");

        var expectedScaleX = (expectedWidth - 1.0) / (sourceWidth - 1.0);
        var expectedScaleY = (expectedHeight - 1.0) / (sourceHeight - 1.0);
        var geometry = result.Geometry();
        AssertClose(expectedScaleX, geometry.HOriginalToRectified[0][0], 1e-8, "H scale X");
        AssertClose(expectedScaleY, geometry.HOriginalToRectified[1][1], 1e-8, "H scale Y");
        AssertClose(1.0 / expectedScaleX, geometry.HRectifiedToOriginal[0][0], 1e-8, "H inverse scale X");
        AssertClose(1.0 / expectedScaleY, geometry.HRectifiedToOriginal[1][1], 1e-8, "H inverse scale Y");
        AssertZeroOffDiagonalAndTranslation(geometry.HOriginalToRectified, "H original-to-rectified");
        AssertZeroOffDiagonalAndTranslation(geometry.HRectifiedToOriginal, "H rectified-to-original");

        var rectifiedBox = new[]
        {
            (float)((expectedWidth - 1) * 0.10),
            (float)((expectedHeight - 1) * 0.20),
            (float)((expectedWidth - 1) * 0.80),
            (float)((expectedHeight - 1) * 0.90),
        };
        var expectedSourceBox = new[]
        {
            (float)(rectifiedBox[0] / expectedScaleX),
            (float)(rectifiedBox[1] / expectedScaleY),
            (float)(rectifiedBox[2] / expectedScaleX),
            (float)(rectifiedBox[3] / expectedScaleY),
        };
        AssertBoxClose(expectedSourceBox, result.ProjectBoxToSource(rectifiedBox), 1e-3f, "scaled box");
    }

    private static Image<Rgb24> CreatePattern(int width, int height)
    {
        var image = new Image<Rgb24>(width, height);
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < width; x++)
                {
                    row[x] = new Rgb24(
                        (byte)((x * 17 + y * 29 + 3) & 255),
                        (byte)((x * 31 + y * 11 + 71) & 255),
                        (byte)((x * 7 + y * 43 + 149) & 255));
                }
            }
        });
        return image;
    }

    private static byte[] PixelBytes(Image<Rgb24> image)
    {
        var bytes = new byte[checked(image.Width * image.Height * 3)];
        var offset = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < image.Height; y++)
            {
                foreach (var pixel in accessor.GetRowSpan(y))
                {
                    bytes[offset++] = pixel.R;
                    bytes[offset++] = pixel.G;
                    bytes[offset++] = pixel.B;
                }
            }
        });
        return bytes;
    }

    private static void AssertIdentity(double[][] matrix, string label)
    {
        for (var row = 0; row < 3; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                AssertClose(row == column ? 1.0 : 0.0, matrix[row][column], 0.0, $"{label}[{row},{column}]");
            }
        }
    }

    private static void AssertZeroOffDiagonalAndTranslation(double[][] matrix, string label)
    {
        AssertClose(0.0, matrix[0][1], 1e-8, label + " m01");
        AssertClose(0.0, matrix[0][2], 1e-8, label + " m02");
        AssertClose(0.0, matrix[1][0], 1e-8, label + " m10");
        AssertClose(0.0, matrix[1][2], 1e-8, label + " m12");
        AssertClose(0.0, matrix[2][0], 1e-8, label + " m20");
        AssertClose(0.0, matrix[2][1], 1e-8, label + " m21");
        AssertClose(1.0, matrix[2][2], 1e-8, label + " m22");
    }

    private static void AssertBoxClose(float[] expected, float[] actual, float tolerance, string label)
    {
        AssertEqual(expected.Length, actual.Length, label + " length");
        for (var index = 0; index < expected.Length; index++)
        {
            if (Math.Abs(expected[index] - actual[index]) > tolerance)
            {
                throw new InvalidOperationException($"{label}[{index}]: expected {expected[index]}, got {actual[index]}");
            }
        }
    }

    private static void AssertClose(double expected, double actual, double tolerance, string label)
    {
        if (Math.Abs(expected - actual) > tolerance)
        {
            throw new InvalidOperationException($"{label}: expected {expected}, got {actual}");
        }
    }

    private static void AssertSequenceEqual(byte[] expected, byte[] actual, string label)
    {
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            throw new InvalidOperationException($"{label}: byte sequences differ");
        }
    }

    private static void AssertEqual<T>(T expected, T actual, string label) where T : IEquatable<T>
    {
        if (!expected.Equals(actual))
        {
            throw new InvalidOperationException($"{label}: expected {expected}, got {actual}");
        }
    }

    private static void Assert(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
