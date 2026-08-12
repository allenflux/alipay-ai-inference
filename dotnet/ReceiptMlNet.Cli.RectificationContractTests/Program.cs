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
            VerifyLandscapePixelRotation();
            VerifyToEvenMaxSideRounding();
            VerifyPortraitMaxSideAndBackProjection();
            VerifyLandscapeMaxSideAndBackProjection();
            VerifySquareDoesNotRotate();
            Console.WriteLine("PASS: portrait orientation, full-image rectification dimensions, homographies, pixels, and box back-projection match the Python contract.");
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
        AssertEqual(
            "RGB_passthrough_to_paddle_v2",
            PaddleOcrImageOps.InputColorOrderContract,
            "Paddle RGB byte-order contract token");
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
        AssertEqual(0, result.Geometry().RotationDegrees, "none mode rotation");
        AssertBoxClose([1.25f, 2.5f, 11.0f, 6.0f], result.ProjectBoxToSource([1.25f, 2.5f, 11.0f, 6.0f]), 0.0f, "none box");
    }

    private static void VerifySmallImageIdentityWarp()
    {
        using var source = CreatePattern(7, 11);
        var expectedPixels = PixelBytes(source);
        using var result = ReceiptRectifier.Rectify(source, ReceiptRectifier.MaxSide1600Mode);
        Assert(!ReferenceEquals(source, result.Image), "max-side-1600 must execute WarpPerspective even below the limit");
        AssertEqual(7, result.Image.Width, "small warp width");
        AssertEqual(11, result.Image.Height, "small warp height");
        AssertSequenceEqual(expectedPixels, PixelBytes(result.Image), "small identity-warp pixels");
        var geometry = result.Geometry();
        AssertIdentity(geometry.HOriginalToRectified, "small H_original_to_rectified");
        AssertIdentity(geometry.HRectifiedToOriginal, "small H_rectified_to_original");
        AssertEqual(0, geometry.RotationDegrees, "small portrait rotation");
        AssertEqual(ReceiptRectifier.MaxSide1600Mode, geometry.Rectification, "small mode");
        Assert(!geometry.ScreenDetected, "full-image mode must not claim screen detection");
    }

    private static void VerifyLandscapePixelRotation()
    {
        using var source = CreatePattern(3, 2);
        using var expected = new Image<Rgb24>(2, 3);
        expected[0, 0] = source[0, 1];
        expected[1, 0] = source[0, 0];
        expected[0, 1] = source[1, 1];
        expected[1, 1] = source[1, 0];
        expected[0, 2] = source[2, 1];
        expected[1, 2] = source[2, 0];

        using var result = ReceiptRectifier.Rectify(source, ReceiptRectifier.MaxSide1600Mode);
        AssertEqual(2, result.Image.Width, "rotated pixel width");
        AssertEqual(3, result.Image.Height, "rotated pixel height");
        AssertEqual(90, result.Geometry().RotationDegrees, "landscape pixel rotation");
        AssertSequenceEqual(PixelBytes(expected), PixelBytes(result.Image), "clockwise rotated pixels");
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
        const int sourceWidth = 2556;
        const int sourceHeight = 1179;
        const int expectedWidth = 738;
        const int expectedHeight = 1600;
        using var source = CreatePattern(sourceWidth, sourceHeight);
        using var result = ReceiptRectifier.Rectify(source, ReceiptRectifier.MaxSide1600Mode);
        AssertEqual(expectedWidth, result.Image.Width, "landscape scaled width");
        AssertEqual(expectedHeight, result.Image.Height, "landscape scaled height");

        var scaleX = (expectedWidth - 1.0) / (sourceHeight - 1.0);
        var scaleY = (expectedHeight - 1.0) / (sourceWidth - 1.0);
        var geometry = result.Geometry();
        AssertEqual(90, geometry.RotationDegrees, "landscape rotation");
        AssertClose(0.0, geometry.HOriginalToRectified[0][0], 1e-8, "landscape H m00");
        AssertClose(-scaleX, geometry.HOriginalToRectified[0][1], 1e-8, "landscape H m01");
        AssertClose(expectedWidth - 1.0, geometry.HOriginalToRectified[0][2], 1e-8, "landscape H m02");
        AssertClose(scaleY, geometry.HOriginalToRectified[1][0], 1e-8, "landscape H m10");
        AssertClose(0.0, geometry.HOriginalToRectified[1][1], 1e-8, "landscape H m11");
        AssertClose(0.0, geometry.HOriginalToRectified[1][2], 1e-8, "landscape H m12");
        AssertClose(0.0, geometry.HRectifiedToOriginal[0][0], 1e-8, "landscape inverse m00");
        AssertClose(1.0 / scaleY, geometry.HRectifiedToOriginal[0][1], 1e-8, "landscape inverse m01");
        AssertClose(0.0, geometry.HRectifiedToOriginal[0][2], 1e-8, "landscape inverse m02");
        AssertClose(-1.0 / scaleX, geometry.HRectifiedToOriginal[1][0], 1e-8, "landscape inverse m10");
        AssertClose(0.0, geometry.HRectifiedToOriginal[1][1], 1e-8, "landscape inverse m11");
        AssertClose(sourceHeight - 1.0, geometry.HRectifiedToOriginal[1][2], 1e-8, "landscape inverse m12");

        var rectifiedBox = new[]
        {
            (float)((expectedWidth - 1) * 0.10),
            (float)((expectedHeight - 1) * 0.20),
            (float)((expectedWidth - 1) * 0.80),
            (float)((expectedHeight - 1) * 0.90),
        };
        var expectedSourceBox = new[]
        {
            (float)(rectifiedBox[1] / scaleY),
            (float)(sourceHeight - 1.0 - rectifiedBox[2] / scaleX),
            (float)(rectifiedBox[3] / scaleY),
            (float)(sourceHeight - 1.0 - rectifiedBox[0] / scaleX),
        };
        AssertBoxClose(expectedSourceBox, result.ProjectBoxToSource(rectifiedBox), 1e-3f, "landscape scaled box");
    }

    private static void VerifySquareDoesNotRotate()
    {
        VerifyScaledCase(sourceWidth: 9, sourceHeight: 9, expectedWidth: 9, expectedHeight: 9);
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
        AssertEqual(0, geometry.RotationDegrees, "portrait/square rotation");
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
