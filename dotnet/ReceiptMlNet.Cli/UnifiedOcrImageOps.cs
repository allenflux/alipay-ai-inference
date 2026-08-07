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
    /// One deterministic recovery crop for a recipient detector box that
    /// contains the right-side value but clipped the visible left anchor.
    /// Keep the original row's vertical extent and right edge, and add only
    /// the horizontal context to its left.  The caller must still run the
    /// complete PP-OCR det/cls/rec pipeline and the strict left-anchor parser.
    /// </summary>
    public static Image<Rgb24>? CropRecipientRowLeftContext(Image<Rgb24> source, float[] box)
    {
        if (box.Length < 4)
        {
            return null;
        }

        var marginX = Math.Max(2.0f, (box[2] - box[0]) * 0.08f);
        var marginY = Math.Max(2.0f, (box[3] - box[1]) * 0.08f);
        var top = Math.Clamp((int)MathF.Floor(box[1] - marginY), 0, source.Height);
        var right = Math.Clamp((int)MathF.Ceiling(box[2] + marginX), 0, source.Width);
        var bottom = Math.Clamp((int)MathF.Ceiling(box[3] + marginY), 0, source.Height);
        if (right <= 0 || bottom <= top)
        {
            return null;
        }

        return source.Clone(context => context.Crop(new Rectangle(0, top, right, bottom - top)));
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
        var values = new float[checked(targetHeight * targetWidth)];
        WriteFieldTensor(image, targetHeight, targetWidth, rightAlign, values, 0, leftCropFraction);
        return values;
    }

    /// <summary>
    /// Write one field tensor directly into a caller-owned fixed-shape ABI
    /// buffer. The requested destination range is overwritten in full.
    /// </summary>
    public static void WriteFieldTensor(
        Image<Rgb24> image,
        int targetHeight,
        int targetWidth,
        bool rightAlign,
        float[] destination,
        int destinationOffset,
        double leftCropFraction = 0.0)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(destination);
        if (targetHeight <= 0 || targetWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Unified OCR target dimensions must be positive");
        }
        if (!double.IsFinite(leftCropFraction) || leftCropFraction < 0.0 || leftCropFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(leftCropFraction), "Unified OCR left crop fraction must be in [0, 1)");
        }
        var valueCount = checked(targetHeight * targetWidth);
        if (destinationOffset < 0 || destinationOffset > destination.Length - valueCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(destinationOffset),
                "Unified OCR tensor destination does not contain the requested output range");
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
        var leftOffset = rightAlign ? targetWidth - resizedWidth : (targetWidth - resizedWidth) / 2;
        var topOffset = (targetHeight - resizedHeight) / 2;
        // L8 has no alpha channel. An opaque DrawImage onto a white canvas is
        // therefore exactly a copy at this offset; write it directly and
        // avoid allocating/traversing the intermediate canvas.
        Array.Fill(destination, 1.0f, destinationOffset, valueCount);
        resized.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < resizedHeight; y++)
            {
                var row = accessor.GetRowSpan(y);
                var destinationRow = destinationOffset + (y + topOffset) * targetWidth + leftOffset;
                for (var x = 0; x < row.Length; x++)
                {
                    destination[destinationRow + x] = row[x].PackedValue / 255.0f;
                }
            }
        });
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
