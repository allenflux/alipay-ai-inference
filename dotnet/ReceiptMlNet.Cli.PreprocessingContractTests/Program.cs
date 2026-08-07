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
            var reusableDetectorBuffer = new float[ImagePipeline.DetectorTensorLength];
            var reusableStatusbarBuffer = new float[ImagePipeline.StatusbarTensorLength];
            VerifyCase(7, 11, reusableDetectorBuffer, reusableStatusbarBuffer);
            VerifyCase(13, 5, reusableDetectorBuffer, reusableStatusbarBuffer);
            VerifyCase(9, 16, reusableDetectorBuffer, reusableStatusbarBuffer);
            VerifyCase(1179, 2556, reusableDetectorBuffer, reusableStatusbarBuffer);
            VerifyDetectorDestinationLengthContract();
            VerifyStatusbarDestinationLengthContract();
            VerifyRecipientTrimUsesPythonDoubleToEven();
            VerifyUnifiedFieldTensorReuse();
            Console.WriteLine("PASS: detector/statusbar/unified OCR preprocessing and reusable buffers are bit-exact against the legacy implementation.");
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

    private static void VerifyCase(
        int width,
        int height,
        float[] reusableDetectorBuffer,
        float[] reusableStatusbarBuffer)
    {
        using var source = CreatePattern(width, height);

        var expectedDetector = LegacyPrepareDetectorInput(source);
        var actualDetector = ImagePipeline.PrepareDetectorInput(source);
        AssertDetectorInputEqual(expectedDetector, actualDetector, $"allocated detector {width}x{height}");

        // Poison every element so bit equality proves both padding clearing and
        // complete RGB-plane writes when this buffer is reused across shapes.
        Array.Fill(
            reusableDetectorBuffer,
            BitConverter.Int32BitsToSingle(unchecked((int)0x7FC12345)));
        var reusedDetector = ImagePipeline.PrepareDetectorInput(source, reusableDetectorBuffer);
        Assert(
            ReferenceEquals(reusableDetectorBuffer, reusedDetector.Tensor),
            "destination overload must return the caller-owned detector buffer");
        AssertDetectorInputEqual(expectedDetector, reusedDetector, $"reused detector {width}x{height}");

        var expectedStatusbar = LegacyPrepareStatusbarInput(source);
        var actualStatusbar = ImagePipeline.PrepareStatusbarInput(source);
        AssertArrayBitsEqual(expectedStatusbar, actualStatusbar, $"allocated statusbar {width}x{height}");

        Array.Fill(
            reusableStatusbarBuffer,
            BitConverter.Int32BitsToSingle(unchecked((int)0x7FC12345)));
        var reusedStatusbar = ImagePipeline.PrepareStatusbarInput(source, reusableStatusbarBuffer);
        Assert(
            ReferenceEquals(reusableStatusbarBuffer, reusedStatusbar),
            "destination overload must return the caller-owned status-bar buffer");
        AssertArrayBitsEqual(expectedStatusbar, reusedStatusbar, $"reused statusbar {width}x{height}");
    }

    private static void VerifyDetectorDestinationLengthContract()
    {
        using var source = CreatePattern(3, 5);
        AssertDetectorDestinationRejected(source, Array.Empty<float>(), "short detector destination");
        AssertDetectorDestinationRejected(
            source,
            new float[ImagePipeline.DetectorTensorLength + 1],
            "long detector destination");
        try
        {
            ImagePipeline.PrepareDetectorInput(source, null!);
        }
        catch (ArgumentNullException error) when (error.ParamName == "destination")
        {
            return;
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                $"null detector destination raised {error.GetType().Name} instead of ArgumentNullException",
                error);
        }
        throw new InvalidOperationException("null detector destination was accepted");
    }

    private static void AssertDetectorDestinationRejected(
        Image<Rgb24> source,
        float[] destination,
        string label)
    {
        try
        {
            ImagePipeline.PrepareDetectorInput(source, destination);
        }
        catch (ArgumentException error) when (error.ParamName == "destination")
        {
            return;
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                $"{label} raised {error.GetType().Name} instead of ArgumentException",
                error);
        }
        throw new InvalidOperationException($"{label} was accepted");
    }

    private static void VerifyStatusbarDestinationLengthContract()
    {
        using var source = CreatePattern(3, 5);
        AssertStatusbarDestinationRejected(source, Array.Empty<float>(), "short status-bar destination");
        AssertStatusbarDestinationRejected(
            source,
            new float[ImagePipeline.StatusbarTensorLength + 1],
            "long status-bar destination");
        try
        {
            ImagePipeline.PrepareStatusbarInput(source, null!);
        }
        catch (ArgumentNullException error) when (error.ParamName == "destination")
        {
            return;
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                $"null status-bar destination raised {error.GetType().Name} instead of ArgumentNullException",
                error);
        }
        throw new InvalidOperationException("null status-bar destination was accepted");
    }

    private static void AssertStatusbarDestinationRejected(
        Image<Rgb24> source,
        float[] destination,
        string label)
    {
        try
        {
            ImagePipeline.PrepareStatusbarInput(source, destination);
        }
        catch (ArgumentException error) when (error.ParamName == "destination")
        {
            return;
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                $"{label} raised {error.GetType().Name} instead of ArgumentException",
                error);
        }
        throw new InvalidOperationException($"{label} was accepted");
    }

    private static void AssertDetectorInputEqual(
        DetectorInputTensor expected,
        DetectorInputTensor actual,
        string label)
    {
        AssertEqual(expected.SourceWidth, actual.SourceWidth, label + " source width");
        AssertEqual(expected.SourceHeight, actual.SourceHeight, label + " source height");
        AssertFloatBitsEqual(expected.ScaleX, actual.ScaleX, label + " scale X");
        AssertFloatBitsEqual(expected.ScaleY, actual.ScaleY, label + " scale Y");
        AssertEqual(expected.OffsetX, actual.OffsetX, label + " offset X");
        AssertEqual(expected.OffsetY, actual.OffsetY, label + " offset Y");
        AssertArrayBitsEqual(expected.Tensor, actual.Tensor, label + " tensor");
    }

    private static void VerifyUnifiedFieldTensorReuse()
    {
        VerifyUnifiedFieldCase(13, 5, 11, 19, rightAlign: true, leftCropFraction: 0.0);
        VerifyUnifiedFieldCase(7, 17, 13, 23, rightAlign: false, leftCropFraction: 0.0);
        VerifyUnifiedFieldCase(55, 9, 16, 31, rightAlign: false, leftCropFraction: 0.30);

        using var source = CreatePattern(7, 11);
        var required = 5 * 9;
        AssertUnifiedDestinationRejected(source, 5, 9, new float[required - 1], 0, "short unified destination");
        AssertUnifiedDestinationRejected(source, 5, 9, new float[required], 1, "offset unified destination");
        try
        {
            UnifiedOcrImageOps.WriteFieldTensor(source, 5, 9, true, null!, 0);
        }
        catch (ArgumentNullException error) when (error.ParamName == "destination")
        {
            return;
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                $"null unified destination raised {error.GetType().Name} instead of ArgumentNullException",
                error);
        }
        throw new InvalidOperationException("null unified destination was accepted");
    }

    private static void VerifyUnifiedFieldCase(
        int width,
        int height,
        int targetHeight,
        int targetWidth,
        bool rightAlign,
        double leftCropFraction)
    {
        using var source = CreatePattern(width, height);
        var expected = LegacyPrepareFieldTensor(source, targetHeight, targetWidth, rightAlign, leftCropFraction);
        var allocated = UnifiedOcrImageOps.PrepareFieldTensor(
            source,
            targetHeight,
            targetWidth,
            rightAlign,
            leftCropFraction);
        AssertArrayBitsEqual(expected, allocated, $"allocated unified {width}x{height}");

        var sentinel = BitConverter.Int32BitsToSingle(unchecked((int)0x7FC12345));
        var destination = Enumerable.Repeat(sentinel, expected.Length + 6).ToArray();
        UnifiedOcrImageOps.WriteFieldTensor(
            source,
            targetHeight,
            targetWidth,
            rightAlign,
            destination,
            destinationOffset: 3,
            leftCropFraction: leftCropFraction);
        AssertArrayBitsEqual(expected, destination.AsSpan(3, expected.Length).ToArray(), $"reused unified {width}x{height}");
        for (var index = 0; index < 3; index++)
        {
            AssertFloatBitsEqual(sentinel, destination[index], "unified destination prefix");
            AssertFloatBitsEqual(sentinel, destination[destination.Length - 1 - index], "unified destination suffix");
        }
    }

    private static void AssertUnifiedDestinationRejected(
        Image<Rgb24> source,
        int targetHeight,
        int targetWidth,
        float[] destination,
        int destinationOffset,
        string label)
    {
        try
        {
            UnifiedOcrImageOps.WriteFieldTensor(
                source,
                targetHeight,
                targetWidth,
                true,
                destination,
                destinationOffset);
        }
        catch (ArgumentOutOfRangeException error) when (error.ParamName == "destinationOffset")
        {
            return;
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                $"{label} raised {error.GetType().Name} instead of ArgumentOutOfRangeException",
                error);
        }
        throw new InvalidOperationException($"{label} was accepted");
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

    private static float[] LegacyPrepareFieldTensor(
        Image<Rgb24> image,
        int targetHeight,
        int targetWidth,
        bool rightAlign,
        double leftCropFraction)
    {
        using var grayscale = new Image<L8>(image.Width, image.Height);
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                grayscale[x, y] = new L8((byte)Math.Clamp(
                    (int)Math.Round(
                        pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114,
                        MidpointRounding.ToEven),
                    byte.MinValue,
                    byte.MaxValue));
            }
        }
        if (leftCropFraction > 0.0)
        {
            var left = Math.Min(
                grayscale.Width - 1,
                Math.Max(0, (int)Math.Round(grayscale.Width * leftCropFraction, MidpointRounding.ToEven)));
            grayscale.Mutate(context => context.Crop(new Rectangle(left, 0, grayscale.Width - left, grayscale.Height)));
        }

        var scale = Math.Min((float)targetWidth / grayscale.Width, (float)targetHeight / grayscale.Height);
        var resizedWidth = Math.Clamp(
            (int)Math.Round(grayscale.Width * scale, MidpointRounding.ToEven),
            1,
            targetWidth);
        var resizedHeight = Math.Clamp(
            (int)Math.Round(grayscale.Height * scale, MidpointRounding.ToEven),
            1,
            targetHeight);
        using var resized = grayscale.Clone(context =>
            context.Resize(resizedWidth, resizedHeight, KnownResamplers.Triangle));
        using var canvas = new Image<L8>(targetWidth, targetHeight, new L8(255));
        var leftOffset = rightAlign ? targetWidth - resizedWidth : (targetWidth - resizedWidth) / 2;
        var topOffset = (targetHeight - resizedHeight) / 2;
        canvas.Mutate(context => context.DrawImage(resized, new Point(leftOffset, topOffset), 1.0f));

        var values = new float[targetHeight * targetWidth];
        for (var y = 0; y < targetHeight; y++)
        {
            for (var x = 0; x < targetWidth; x++)
            {
                values[y * targetWidth + x] = canvas[x, y].PackedValue / 255.0f;
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

    private static void Assert(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
