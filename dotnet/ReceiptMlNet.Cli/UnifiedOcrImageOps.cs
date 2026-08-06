using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

/// <summary>
/// Image preprocessing for the static v12 unified-reader ABI.  The tensor
/// layout is deliberately simple and explicit: grayscale NCHW with white
/// letterboxing, matching the Python delivery contract.
/// </summary>
internal static class UnifiedOcrImageOps
{
    public static Image<Rgb24>? CropFieldWithMargin(Image<Rgb24> source, float[] box)
    {
        if (box.Length < 4)
        {
            return null;
        }

        var marginX = Math.Max(2.0f, (box[2] - box[0]) * 0.08f);
        var marginY = Math.Max(2.0f, (box[3] - box[1]) * 0.08f);
        var left = Math.Clamp((int)MathF.Floor(box[0] - marginX), 0, source.Width);
        var top = Math.Clamp((int)MathF.Floor(box[1] - marginY), 0, source.Height);
        var right = Math.Clamp((int)MathF.Ceiling(box[2] + marginX), 0, source.Width);
        var bottom = Math.Clamp((int)MathF.Ceiling(box[3] + marginY), 0, source.Height);
        if (right <= left || bottom <= top)
        {
            return null;
        }

        return source.Clone(context => context.Crop(new Rectangle(left, top, right - left, bottom - top)));
    }

    /// <summary>
    /// Build one [1,H,W] grayscale tensor. A right-aligned canvas is used by
    /// amount/time/payment slots; transfer-status and the v12 recipient view
    /// use a centered canvas. Missing slots are represented by an all-white
    /// tensor by the caller and are never decoded as a delivered value.
    /// </summary>
    public static float[] PrepareFieldTensor(
        Image<Rgb24> image,
        int targetHeight,
        int targetWidth,
        bool rightAlign,
        double leftCropFraction = 0.0)
    {
        if (targetHeight <= 0 || targetWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Unified OCR target dimensions must be positive");
        }
        if (!double.IsFinite(leftCropFraction) || leftCropFraction < 0.0 || leftCropFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(leftCropFraction), "Unified OCR left crop fraction must be in [0, 1)");
        }

        using var grayscale = ToGrayscale(image);
        if (leftCropFraction > 0.0)
        {
            var left = LeftTrimPixels(grayscale.Width, leftCropFraction);
            grayscale.Mutate(context => context.Crop(new Rectangle(left, 0, grayscale.Width - left, grayscale.Height)));
        }

        var scale = Math.Min((float)targetWidth / grayscale.Width, (float)targetHeight / grayscale.Height);
        var resizedWidth = Math.Clamp((int)Math.Round(grayscale.Width * scale, MidpointRounding.ToEven), 1, targetWidth);
        var resizedHeight = Math.Clamp((int)Math.Round(grayscale.Height * scale, MidpointRounding.ToEven), 1, targetHeight);
        using var resized = grayscale.Clone(context => context.Resize(resizedWidth, resizedHeight, KnownResamplers.Triangle));
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

    internal static int LeftTrimPixels(int width, double leftCropFraction)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }
        if (!double.IsFinite(leftCropFraction) || leftCropFraction < 0.0 || leftCropFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(leftCropFraction));
        }
        return Math.Min(
            width - 1,
            Math.Max(0, (int)Math.Round(width * leftCropFraction, MidpointRounding.ToEven)));
    }

    private static Image<L8> ToGrayscale(Image<Rgb24> source)
    {
        var grayscale = new Image<L8>(source.Width, source.Height);
        for (var y = 0; y < source.Height; y++)
        {
            for (var x = 0; x < source.Width; x++)
            {
                var pixel = source[x, y];
                // Pillow Image.convert("L")'s standard RGB luma weights.
                grayscale[x, y] = new L8((byte)Math.Clamp(
                    (int)Math.Round(
                        pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114,
                        MidpointRounding.ToEven),
                    byte.MinValue,
                    byte.MaxValue));
            }
        }
        return grayscale;
    }
}
