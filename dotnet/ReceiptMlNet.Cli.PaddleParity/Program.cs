using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

internal static class Program
{
    private static readonly HashSet<string> ImageExtensions =
        new([".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"], StringComparer.OrdinalIgnoreCase);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        WriteIndented = true,
    };

    private static int Main(string[] args)
    {
        try
        {
            var options = Options.Parse(args);
            Run(options);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Paddle .NET parity failed: {error.Message}");
            return 1;
        }
    }

    private static void Run(Options options)
    {
        var output = Path.GetFullPath(options.Output);
        if (Directory.Exists(output) || File.Exists(output))
        {
            throw new InvalidOperationException($"Refusing to overwrite .NET Paddle parity output: {output}");
        }
        var images = EnumerateImages(options.Input).ToArray();
        if (images.Length == 0)
        {
            throw new InvalidOperationException($"No parity images found under {options.Input}");
        }

        var bundle = PaddleOcrDeliveryBundle.LoadAndVerify(options.Bundle);
        var device = DeviceSetting.Parse("cpu");
        using var engine = new PaddleOcrEngine(bundle, device);
        if (!string.Equals(engine.ExecutionProvider, "cpu", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Paddle parity must use CPU, got {engine.ExecutionProvider}");
        }

        var parent = Path.GetDirectoryName(output)
            ?? throw new InvalidOperationException("Parity output must have a parent directory");
        Directory.CreateDirectory(parent);
        var stage = Path.Combine(parent, $".{Path.GetFileName(output)}.{Guid.NewGuid():N}");
        Directory.CreateDirectory(stage);
        try
        {
            var records = new List<ParityRecord>(images.Length);
            var latencies = new List<double>(images.Length);
            foreach (var path in images)
            {
                using var image = Image.Load<Rgb24>(path);
                var stopwatch = Stopwatch.StartNew();
                var read = engine.Recognize(image);
                stopwatch.Stop();
                var elapsed = Math.Round(stopwatch.Elapsed.TotalMilliseconds, 4);
                latencies.Add(elapsed);
                records.Add(new ParityRecord(
                    Path.GetFullPath(path),
                    read.Text,
                    PaddleRecipientValueParser.Parse(read.Text),
                    read.Confidence,
                    read.Lines.Select(line => new ParityLine(line.Text, line.Confidence)).ToArray(),
                    elapsed));
            }

            var recordsPath = Path.Combine(stage, "records.jsonl");
            using (var writer = new StreamWriter(recordsPath, append: false, encoding: new UTF8Encoding(false)))
            {
                foreach (var record in records)
                {
                    writer.WriteLine(JsonSerializer.Serialize(record, JsonOptions));
                }
            }
            var sorted = latencies.OrderBy(value => value).ToArray();
            var summary = new ParitySummary(
                1,
                "receipt_ppocr_dotnet_cpu_parity_v1",
                records.Count,
                "cpu",
                bundle.ContractSha256,
                Math.Round(Percentile(sorted, 0.50), 4),
                Math.Round(Percentile(sorted, 0.95), 4));
            File.WriteAllText(
                Path.Combine(stage, "summary.json"),
                JsonSerializer.Serialize(summary, JsonOptions) + Environment.NewLine,
                new UTF8Encoding(false));
            Directory.Move(stage, output);
        }
        catch
        {
            Directory.Delete(stage, recursive: true);
            throw;
        }
        Console.WriteLine($"Wrote {images.Length} .NET PP-OCR CPU parity result(s) to {output}");
    }

    private static IEnumerable<string> EnumerateImages(string input)
    {
        var path = Path.GetFullPath(input);
        if (File.Exists(path))
        {
            if (!ImageExtensions.Contains(Path.GetExtension(path)))
            {
                throw new InvalidOperationException($"Unsupported parity image: {path}");
            }
            return [path];
        }
        if (!Directory.Exists(path))
        {
            throw new InvalidOperationException($"Parity input does not exist: {path}");
        }
        return Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories)
            .Where(file => ImageExtensions.Contains(Path.GetExtension(file)))
            .OrderBy(file => file, StringComparer.Ordinal);
    }

    private static double Percentile(IReadOnlyList<double> sorted, double quantile)
    {
        var position = (sorted.Count - 1) * quantile;
        var lower = (int)Math.Floor(position);
        var upper = (int)Math.Ceiling(position);
        return lower == upper
            ? sorted[lower]
            : sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private sealed record Options(string Bundle, string Input, string Output)
    {
        public static Options Parse(string[] args)
        {
            string? bundle = null;
            string? input = null;
            string? output = null;
            for (var index = 0; index < args.Length; index++)
            {
                var value = index + 1 < args.Length ? args[index + 1] : null;
                switch (args[index])
                {
                    case "--bundle": bundle = value; index++; break;
                    case "--input": input = value; index++; break;
                    case "--output": output = value; index++; break;
                    default: throw new InvalidOperationException($"Unknown or incomplete argument: {args[index]}");
                }
            }
            if (string.IsNullOrWhiteSpace(bundle) || string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            {
                throw new InvalidOperationException("Usage: --bundle <delivery> --input <crop-or-dir> --output <fresh-dir>");
            }
            return new Options(bundle, input, output);
        }
    }

    private sealed record ParityLine(string Text, float Confidence);
    private sealed record ParityRecord(
        string Source,
        string RawText,
        string? CandidateAnchoredValue,
        float? Confidence,
        IReadOnlyList<ParityLine> Lines,
        double ElapsedMs);
    private sealed record ParitySummary(
        int SchemaVersion,
        string Kind,
        int Records,
        string ExecutionProvider,
        string BundleContractSha256,
        double P50Ms,
        double P95Ms);
}
