using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

internal static class Program
{
    private static int Main()
    {
        try
        {
            // Tiny portrait/landscape inputs exercise both letterbox axes and
            // rounding. The production-size input catches row/plane-offset
            // mistakes at the actual detector tensor dimensions.
            VerifyCase(7, 11);
            VerifyCase(13, 5);
            VerifyCase(9, 16);
            VerifyCase(1179, 2556);
            VerifyRecipientTrimUsesPythonDoubleToEven();
            Console.WriteLine("PASS: detector/statusbar preprocessing is bit-exact against the legacy implementation.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static void VerifyRecipientTrimUsesPythonDoubleToEven()
    {
        // Python round(55 * 0.30) sees the exact JSON double computation at
        // 16.5 and chooses the even integer 16.  Parsing 0.30 as float first
        // produces 16.500000655... and incorrectly trims 17 pixels.
        AssertEqual(16, UnifiedOcrImageOps.LeftTrimPixels(55, 0.30), "recipient trim 16.5 to even");
        AssertEqual(20, UnifiedOcrImageOps.LeftTrimPixels(65, 0.30), "recipient trim 19.5 to even");
    }

    private static void VerifyCase(int width, int height)
    {
        using var source = CreatePattern(width, height);

        var expectedDetector = LegacyPrepareDetectorInput(source);
        var actualDetector = ImagePipeline.PrepareDetectorInput(source);
        AssertEqual(expectedDetector.SourceWidth, actualDetector.SourceWidth, "detector source width");
        AssertEqual(expectedDetector.SourceHeight, actualDetector.SourceHeight, "detector source height");
        AssertFloatBitsEqual(expectedDetector.ScaleX, actualDetector.ScaleX, "detector scale X");
        AssertFloatBitsEqual(expectedDetector.ScaleY, actualDetector.ScaleY, "detector scale Y");
        AssertEqual(expectedDetector.OffsetX, actualDetector.OffsetX, "detector offset X");
        AssertEqual(expectedDetector.OffsetY, actualDetector.OffsetY, "detector offset Y");
        AssertArrayBitsEqual(expectedDetector.Tensor, actualDetector.Tensor, $"detector {width}x{height}");

        var expectedStatusbar = LegacyPrepareStatusbarInput(source);
        var actualStatusbar = ImagePipeline.PrepareStatusbarInput(source);
        AssertArrayBitsEqual(expectedStatusbar, actualStatusbar, $"statusbar {width}x{height}");
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
                    // Non-symmetric channels reveal CHW plane swaps; primes
                    // make neighbouring rows/pixels unlikely to match.
                    row[x] = new Rgb24(
                        (byte)((x * 17 + y * 29 + 3) & 255),
                        (byte)((x * 31 + y * 11 + 71) & 255),
                        (byte)((x * 7 + y * 43 + 149) & 255));
                }
            }
        });
        return image;
    }

    // Frozen reference: this is the implementation used before the row-span
    // optimisation. Keep it independent so the contract test detects changes
    // to padding, resampling, CHW placement or normalisation arithmetic.
    private static DetectorInputTensor LegacyPrepareDetectorInput(Image<Rgb24> source)
    {
        var scale = Math.Min(
            (float)ImagePipeline.DetectorWidth / source.Width,
            (float)ImagePipeline.DetectorHeight / source.Height);
        var resizedWidth = Math.Clamp(
            (int)Math.Round(source.Width * scale, MidpointRounding.ToEven),
            1,
            ImagePipeline.DetectorWidth);
        var resizedHeight = Math.Clamp(
            (int)Math.Round(source.Height * scale, MidpointRounding.ToEven),
            1,
            ImagePipeline.DetectorHeight);
        var left = (ImagePipeline.DetectorWidth - resizedWidth) / 2;
        var top = (ImagePipeline.DetectorHeight - resizedHeight) / 2;
        using var resized = source.Clone(context =>
            context.Resize(resizedWidth, resizedHeight, KnownResamplers.Triangle));
        using var canvas = new Image<Rgb24>(ImagePipeline.DetectorWidth, ImagePipeline.DetectorHeight);
        canvas.Mutate(context => context.DrawImage(resized, new Point(left, top), 1.0f));

        var values = new float[3 * ImagePipeline.DetectorHeight * ImagePipeline.DetectorWidth];
        var plane = ImagePipeline.DetectorHeight * ImagePipeline.DetectorWidth;
        for (var y = 0; y < ImagePipeline.DetectorHeight; y++)
        {
            for (var x = 0; x < ImagePipeline.DetectorWidth; x++)
            {
                var pixel = canvas[x, y];
                var offset = y * ImagePipeline.DetectorWidth + x;
                values[offset] = pixel.R / 255.0f;
                values[plane + offset] = pixel.G / 255.0f;
                values[2 * plane + offset] = pixel.B / 255.0f;
            }
        }

        return new DetectorInputTensor(
            values,
            source.Width,
            source.Height,
            (float)resizedWidth / source.Width,
            (float)resizedHeight / source.Height,
            left,
            top);
    }

    private static float[] LegacyPrepareStatusbarInput(Image<Rgb24> source)
    {
        var stripHeight = Math.Max(1, (int)Math.Round(source.Height * 0.08, MidpointRounding.ToEven));
        using var strip = source.Clone(context =>
            context.Crop(new Rectangle(0, 0, source.Width, stripHeight)));
        using var canvas = strip.Clone(context =>
            context.Resize(ImagePipeline.StatusbarWidth, ImagePipeline.StatusbarHeight, KnownResamplers.Bicubic));
        var values = new float[3 * ImagePipeline.StatusbarHeight * ImagePipeline.StatusbarWidth];
        var plane = ImagePipeline.StatusbarHeight * ImagePipeline.StatusbarWidth;
        for (var y = 0; y < ImagePipeline.StatusbarHeight; y++)
        {
            for (var x = 0; x < ImagePipeline.StatusbarWidth; x++)
            {
                var pixel = canvas[x, y];
                var offset = y * ImagePipeline.StatusbarWidth + x;
                values[offset] = (pixel.R / 255.0f - 0.485f) / 0.229f;
                values[plane + offset] = (pixel.G / 255.0f - 0.456f) / 0.224f;
                values[2 * plane + offset] = (pixel.B / 255.0f - 0.406f) / 0.225f;
            }
        }
        return values;
    }

    private static void AssertArrayBitsEqual(float[] expected, float[] actual, string label)
    {
        AssertEqual(expected.Length, actual.Length, $"{label} length");
        for (var index = 0; index < expected.Length; index++)
        {
            var expectedBits = BitConverter.SingleToInt32Bits(expected[index]);
            var actualBits = BitConverter.SingleToInt32Bits(actual[index]);
            if (expectedBits != actualBits)
            {
                throw new InvalidOperationException(
                    $"{label} differs at index {index}: " +
                    $"expected {expected[index]} (0x{expectedBits:X8}), " +
                    $"actual {actual[index]} (0x{actualBits:X8})");
            }
        }
    }

    private static void AssertFloatBitsEqual(float expected, float actual, string label)
    {
        var expectedBits = BitConverter.SingleToInt32Bits(expected);
        var actualBits = BitConverter.SingleToInt32Bits(actual);
        if (expectedBits != actualBits)
        {
            throw new InvalidOperationException(
                $"{label}: expected {expected} (0x{expectedBits:X8}), actual {actual} (0x{actualBits:X8})");
        }
    }

    private static void AssertEqual(int expected, int actual, string label)
    {
        if (expected != actual)
        {
            throw new InvalidOperationException($"{label}: expected {expected}, actual {actual}");
        }
    }
}
