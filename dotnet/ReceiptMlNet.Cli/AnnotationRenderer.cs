using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using IOPath = System.IO.Path;

internal sealed record AnnotationPaths(string Rectified, string Original)
{
    public static AnnotationPaths ForResultJson(string resultJsonPath)
    {
        var fullResultPath = IOPath.GetFullPath(resultJsonPath);
        var directory = IOPath.GetDirectoryName(fullResultPath)!;
        var stem = IOPath.GetFileNameWithoutExtension(fullResultPath);
        return new AnnotationPaths(
            IOPath.Combine(directory, stem + "_rectified_annotated.jpg"),
            IOPath.Combine(directory, stem + "_original_annotated.jpg"));
    }
}

/// <summary>
/// Renders the same inspection-oriented visual conventions as the Python
/// pipeline: expanded field ellipses, fixed field colors, and a side legend.
/// The ML.NET path currently accepts already-rectified input, so its two
/// compatibility-named JPGs have the same upright source-coordinate content.
/// </summary>
internal static class AnnotationRenderer
{
    private static readonly Dictionary<string, LabelPresentation> Presentations = new(StringComparer.Ordinal)
    {
        ["time"] = new("时间", Color.FromRgb(222, 82, 255), 0),
        ["amount"] = new("金额", Color.FromRgb(255, 80, 80), 1),
        ["transfer_status"] = new("转账状态", Color.FromRgb(255, 210, 0), 2),
        ["recipient_field"] = new("收款方", Color.FromRgb(72, 202, 128), 3),
        ["payment_method_field"] = new("付款方式", Color.FromRgb(80, 160, 255), 4),
    };

    private static readonly Color PanelSeparator = Color.FromRgb(190, 195, 205);
    private static readonly Color TextColor = Color.FromRgb(25, 28, 35);
    private static readonly Color DeviceColor = Color.FromRgb(220, 30, 30);
    private static readonly Color FallbackColor = Color.FromRgb(255, 0, 255);

    public static void RenderAndSave(
        string inputFile,
        IReadOnlyCollection<DetectionResult> detections,
        DeviceResult? device,
        AnnotationPaths paths)
    {
        Directory.CreateDirectory(IOPath.GetDirectoryName(paths.Rectified)!);
        using var source = ImagePipeline.LoadUprightRgb(inputFile);
        using var canvas = Render(source, detections, device);
        var encoder = new JpegEncoder { Quality = 95 };
        canvas.SaveAsJpeg(paths.Rectified, encoder);
        canvas.SaveAsJpeg(paths.Original, encoder);
    }

    private static Image<Rgb24> Render(
        Image<Rgb24> source,
        IReadOnlyCollection<DetectionResult> detections,
        DeviceResult? device)
    {
        var ordered = detections
            .Where(item => item.BboxImage.Length >= 4)
            .OrderBy(item => PresentationFor(item.Label).Order)
            .ThenBy(item => item.Label, StringComparer.Ordinal)
            .ToArray();
        if (ordered.Length == 0 && device is null)
        {
            return source.Clone();
        }

        var lineWidth = Math.Clamp(
            (int)Math.Round(Math.Max(source.Width, source.Height) * 0.002, MidpointRounding.ToEven),
            3,
            7);
        var panelWidth = Math.Clamp(
            (int)Math.Round(source.Width * 0.52, MidpointRounding.ToEven),
            300,
            680);
        var fontSize = Math.Max(
            16,
            Math.Min(30, Math.Min(source.Height / 40, (int)Math.Round(Math.Max(source.Width, source.Height) * 0.012, MidpointRounding.ToEven))));
        var padding = Math.Max(10, fontSize / 2);
        var fonts = SelectFontFamily();
        var font = fonts.Family.CreateFont(fontSize);
        var titleFont = fonts.Family.CreateFont(Math.Min(34, fontSize + 3));

        var canvas = new Image<Rgb24>(source.Width + panelWidth, source.Height, new Rgb24(247, 248, 250));
        canvas.Mutate(context =>
        {
            context.DrawImage(source, new Point(0, 0), 1.0f);
            foreach (var detection in ordered)
            {
                var box = detection.BboxImage;
                var centerX = (box[0] + box[2]) / 2.0f;
                var centerY = (box[1] + box[3]) / 2.0f;
                var ellipseWidth = Math.Max(4.0f, Math.Abs(box[2] - box[0]) + 6.0f);
                var ellipseHeight = Math.Max(4.0f, Math.Abs(box[3] - box[1]) + 6.0f);
                context.Draw(
                    Pens.Solid(PresentationFor(detection.Label).Color, lineWidth),
                    new EllipsePolygon(centerX, centerY, ellipseWidth, ellipseHeight));
            }

            context.Fill(PanelSeparator, new RectangleF(source.Width, 0, Math.Max(2, lineWidth / 2), source.Height));
            var panelLeft = source.Width + padding;
            var panelRight = source.Width + panelWidth - padding;
            var title = fonts.SupportsChinese ? "识别结果" : "Detection results";
            context.DrawText(title, titleFont, TextColor, new PointF(panelLeft, padding));

            var cursorY = padding + fontSize + padding;
            var entryHeight = fontSize + padding * 2;
            for (var index = 0; index < ordered.Length; index++)
            {
                var detection = ordered[index];
                var presentation = PresentationFor(detection.Label);
                var captionLabel = fonts.SupportsChinese ? presentation.DisplayName : detection.Label;
                var ocrText = detection.Ocr is { Text.Length: > 0 }
                    ? $": {Truncate(detection.Ocr.Text, 28)}"
                    : string.Empty;
                var caption = $"{index + 1}. {captionLabel}{ocrText} ({Percent(detection.Score)})";
                if (!DrawCard(context, panelLeft, panelRight, cursorY, entryHeight, padding, lineWidth, presentation.Color, caption, font, TextColor, source.Height))
                {
                    return;
                }
                cursorY += entryHeight + Math.Max(7, padding / 2);
            }

            if (device is not null)
            {
                var deviceCaption = fonts.SupportsChinese
                    ? $"{ordered.Length + 1}. 设备 {device.PlatformCn} ({Percent(device.Confidence)})"
                    : $"{ordered.Length + 1}. Device {device.Platform} ({Percent(device.Confidence)})";
                _ = DrawCard(context, panelLeft, panelRight, cursorY, entryHeight, padding, lineWidth, DeviceColor, deviceCaption, font, DeviceColor, source.Height);
            }
        });
        return canvas;
    }

    private static bool DrawCard(
        IImageProcessingContext context,
        float left,
        float right,
        float top,
        float height,
        int padding,
        int lineWidth,
        Color color,
        string caption,
        Font font,
        Color textColor,
        int imageHeight)
    {
        if (top + height > imageHeight - padding)
        {
            return false;
        }
        var rectangle = new RectangleF(left, top, right - left, height);
        context.Fill(Color.White, rectangle);
        context.Draw(Pens.Solid(color, Math.Max(2, lineWidth / 2)), rectangle);
        var stripeWidth = Math.Max(5, padding / 2);
        context.Fill(color, new RectangleF(left, top, stripeWidth, height));
        context.DrawText(caption, font, textColor, new PointF(left + stripeWidth + padding, top + padding));
        return true;
    }

    private static LabelPresentation PresentationFor(string label)
    {
        return Presentations.TryGetValue(label, out var presentation)
            ? presentation
            : new LabelPresentation(label, FallbackColor, 999);
    }

    private static FontSelection SelectFontFamily()
    {
        foreach (var familyName in new[] { "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun" })
        {
            if (SystemFonts.TryGet(familyName, out var family))
            {
                return new FontSelection(family, true);
            }
        }
        foreach (var familyName in new[] { "Arial", "Segoe UI", "Noto Sans" })
        {
            if (SystemFonts.TryGet(familyName, out var family))
            {
                return new FontSelection(family, false);
            }
        }
        foreach (var family in SystemFonts.Families)
        {
            return new FontSelection(family, false);
        }
        throw new InvalidOperationException("No system font is available for annotation rendering");
    }

    private static string Percent(float value)
    {
        var percent = Math.Clamp((int)Math.Round(value * 100.0f, MidpointRounding.ToEven), 0, 100);
        return $"{percent}%";
    }

    private static string Truncate(string value, int maximumLength)
    {
        return value.Length <= maximumLength ? value : value[..Math.Max(0, maximumLength - 1)] + "…";
    }

    private sealed record LabelPresentation(string DisplayName, Color Color, int Order);
    private sealed record FontSelection(FontFamily Family, bool SupportsChinese);
}
