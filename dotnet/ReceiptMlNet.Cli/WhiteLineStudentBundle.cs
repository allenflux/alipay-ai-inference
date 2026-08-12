using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

/// <summary>
/// Byte-closed delivery bundle for the optional white-document line student.
/// The bundle is deliberately narrower than the general receipt CTC export:
/// it accepts only a model trained for <c>generic_text_line</c> and binds the
/// ONNX, character map, preprocessing ABI and CTC output before ORT opens.
/// </summary>
internal sealed class WhiteLineStudentBundle
{
    public const string ContractKind = "receipt_ocr_ctc_v1";
    public const string FieldKind = "generic_text_line";
    public const string InputName = "image";
    public const string OutputName = "logits";
    public const string Preprocess = "opencv_exact_rgb_gray_letterbox_v1";

    private WhiteLineStudentBundle(
        string deliveryDirectory,
        string modelFileName,
        string charsetFileName,
        string contractFileName,
        byte[] modelBytes,
        byte[] charsetBytes,
        byte[] contractBytes,
        int imageHeight,
        int imageWidth,
        IReadOnlyList<string> characters)
    {
        DeliveryDirectory = deliveryDirectory;
        ModelFileName = modelFileName;
        CharsetFileName = charsetFileName;
        ContractFileName = contractFileName;
        _modelBytes = modelBytes;
        _charsetBytes = charsetBytes;
        _contractBytes = contractBytes;
        ImageHeight = imageHeight;
        ImageWidth = imageWidth;
        Characters = characters;
    }

    private readonly byte[] _modelBytes;
    private readonly byte[] _charsetBytes;
    private readonly byte[] _contractBytes;

    public string DeliveryDirectory { get; }
    public string ModelFileName { get; }
    public string CharsetFileName { get; }
    public string ContractFileName { get; }
    public int ImageHeight { get; }
    public int ImageWidth { get; }
    public IReadOnlyList<string> Characters { get; }
    public string ModelSha256 => Sha256(_modelBytes);
    public string CharsetSha256 => Sha256(_charsetBytes);
    public string ContractSha256 => Sha256(_contractBytes);
    public long ModelSizeBytes => _modelBytes.LongLength;
    public long CharsetSizeBytes => _charsetBytes.LongLength;
    public long ContractSizeBytes => _contractBytes.LongLength;

    internal InferenceSession OpenCpuSession()
    {
        using var options = new SessionOptions();
        options.AppendExecutionProvider_CPU(1);
        return new InferenceSession(_modelBytes, options);
    }

    public static WhiteLineStudentBundle LoadAndVerify(string deliveryDirectory)
    {
        var directory = Path.GetFullPath(deliveryDirectory);
        if (!Directory.Exists(directory))
        {
            throw new UsageException($"White line student bundle directory not found: {directory}");
        }

        var contractPaths = Directory.GetFiles(directory, "*.contract.json", SearchOption.TopDirectoryOnly);
        if (contractPaths.Length != 1)
        {
            throw new UsageException(
                $"White line student bundle must contain exactly one *.contract.json; found {contractPaths.Length}");
        }

        var contractPath = contractPaths[0];
        var contractBytes = File.ReadAllBytes(contractPath);
        try
        {
            using var contractDocument = JsonDocument.Parse(contractBytes);
            var contract = RequireObjectRoot(contractDocument.RootElement, contractPath);
            RequireInteger(contract, "schema_version", 1, contractPath);
            RequireString(contract, "kind", ContractKind, contractPath);

            var modelFileName = RequireLeafFileName(contract, "onnx_file", ".onnx", contractPath);
            var charsetFileName = RequireLeafFileName(contract, "charset_file", ".charset.json", contractPath);
            var modelPath = Path.Combine(directory, modelFileName);
            var charsetPath = Path.Combine(directory, charsetFileName);
            if (!File.Exists(modelPath) || !File.Exists(charsetPath))
            {
                throw ContractError(contractPath, "referenced ONNX or charset artifact is missing");
            }

            var modelBytes = File.ReadAllBytes(modelPath);
            var charsetBytes = File.ReadAllBytes(charsetPath);
            RequireHash(contract, "onnx_sha256", modelBytes, contractPath);
            RequireHash(contract, "charset_sha256", charsetBytes, contractPath);
            RequireExactStringArray(contract, "fields", [FieldKind], contractPath);

            using var charsetDocument = JsonDocument.Parse(charsetBytes);
            var charset = RequireObjectRoot(charsetDocument.RootElement, charsetPath);
            RequireInteger(charset, "schema_version", 1, charsetPath);
            RequireInteger(charset, "blank_index", 0, charsetPath);
            var characters = RequireCharacters(charset, charsetPath);
            RequireHashText(charset, "sha256", string.Concat(characters), charsetPath);

            var input = RequireObject(contract, "input", contractPath);
            RequireString(input, "name", InputName, contractPath);
            RequireString(input, "dtype", "float32", contractPath);
            RequireString(input, "preprocess", Preprocess, contractPath);
            var inputShape = RequireIntArray(input, "shape", 4, contractPath);
            if (inputShape[0] != 1 || inputShape[1] != 1 || inputShape[2] < 16 || inputShape[3] < 64)
            {
                throw ContractError(contractPath, "input.shape must be static [1,1,H>=16,W>=64]");
            }

            var output = RequireObject(contract, "output", contractPath);
            RequireString(output, "name", OutputName, contractPath);
            RequireString(output, "layout", "[time,batch,class]", contractPath);
            RequireString(output, "decoder", "ctc_greedy", contractPath);
            RequireInteger(output, "blank_index", 0, contractPath);
            var outputShape = RequireIntArray(output, "shape", 3, contractPath);
            if (outputShape[0] <= 0 || outputShape[1] != 1 || outputShape[2] != characters.Count + 1)
            {
                throw ContractError(
                    contractPath,
                    "output.shape must be static [time,1,charset+blank]");
            }

            return new WhiteLineStudentBundle(
                directory,
                modelFileName,
                charsetFileName,
                Path.GetFileName(contractPath),
                modelBytes.ToArray(),
                charsetBytes.ToArray(),
                contractBytes.ToArray(),
                inputShape[2],
                inputShape[3],
                characters);
        }
        catch (JsonException error)
        {
            throw new UsageException($"Invalid white line student bundle JSON: {error.Message}");
        }
    }

    internal static byte[] VerifyAndClone(byte[] bytes, long expectedSize, string expectedSha256, string role)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (bytes.LongLength != expectedSize
            || !string.Equals(Sha256(bytes), expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"White line student {role} byte snapshot differs from its verified contract");
        }
        return bytes.ToArray();
    }

    private static JsonElement RequireObjectRoot(JsonElement value, string path)
    {
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw ContractError(path, "document root must be an object");
        }
        return value;
    }

    private static JsonElement RequireObject(JsonElement source, string name, string path)
    {
        if (!source.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Object)
        {
            throw ContractError(path, $"missing or invalid object {name}");
        }
        return value;
    }

    private static string RequireStringValue(JsonElement source, string name, string path)
    {
        if (!source.TryGetProperty(name, out var value)
            || value.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw ContractError(path, $"missing or invalid string {name}");
        }
        return value.GetString()!;
    }

    private static void RequireString(JsonElement source, string name, string expected, string path)
    {
        var actual = RequireStringValue(source, name, path);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw ContractError(path, $"{name} must be '{expected}', found '{actual}'");
        }
    }

    private static void RequireInteger(JsonElement source, string name, int expected, string path)
    {
        if (!source.TryGetProperty(name, out var value)
            || !value.TryGetInt32(out var actual)
            || actual != expected)
        {
            throw ContractError(path, $"{name} must be integer {expected}");
        }
    }

    private static string RequireLeafFileName(JsonElement source, string name, string suffix, string path)
    {
        var value = RequireStringValue(source, name, path);
        if (value.IndexOfAny(['/', '\\']) >= 0
            || !string.Equals(Path.GetFileName(value), value, StringComparison.Ordinal)
            || !value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            throw ContractError(path, $"{name} must be a leaf file name ending in {suffix}");
        }
        return value;
    }

    private static int[] RequireIntArray(JsonElement source, string name, int count, string path)
    {
        if (!source.TryGetProperty(name, out var value)
            || value.ValueKind != JsonValueKind.Array
            || value.GetArrayLength() != count)
        {
            throw ContractError(path, $"{name} must contain exactly {count} integers");
        }
        var output = new int[count];
        for (var index = 0; index < count; index++)
        {
            if (!value[index].TryGetInt32(out output[index]))
            {
                throw ContractError(path, $"{name}[{index}] must be an integer");
            }
        }
        return output;
    }

    private static void RequireExactStringArray(
        JsonElement source,
        string name,
        IReadOnlyList<string> expected,
        string path)
    {
        if (!source.TryGetProperty(name, out var value)
            || value.ValueKind != JsonValueKind.Array
            || value.GetArrayLength() != expected.Count)
        {
            throw ContractError(path, $"{name} must equal [{string.Join(',', expected)}]");
        }
        for (var index = 0; index < expected.Count; index++)
        {
            if (value[index].ValueKind != JsonValueKind.String
                || !string.Equals(value[index].GetString(), expected[index], StringComparison.Ordinal))
            {
                throw ContractError(path, $"{name} must equal [{string.Join(',', expected)}]");
            }
        }
    }

    private static IReadOnlyList<string> RequireCharacters(JsonElement source, string path)
    {
        if (!source.TryGetProperty("characters", out var value)
            || value.ValueKind != JsonValueKind.Array
            || value.GetArrayLength() == 0)
        {
            throw ContractError(path, "characters must be a non-empty array");
        }
        var characters = value.EnumerateArray()
            .Select(item => item.ValueKind == JsonValueKind.String ? item.GetString() : null)
            .ToArray();
        if (characters.Any(item => string.IsNullOrEmpty(item) || item!.EnumerateRunes().Count() != 1)
            || characters.Distinct(StringComparer.Ordinal).Count() != characters.Length)
        {
            throw ContractError(path, "characters must be unique single Unicode code points");
        }
        return Array.AsReadOnly(characters.Select(item => item!).ToArray());
    }

    private static void RequireHash(JsonElement source, string name, byte[] bytes, string path)
    {
        var expected = RequireStringValue(source, name, path);
        if (!string.Equals(expected, Sha256(bytes), StringComparison.OrdinalIgnoreCase))
        {
            throw ContractError(path, $"{name} differs from the referenced artifact bytes");
        }
    }

    private static void RequireHashText(JsonElement source, string name, string value, string path)
    {
        var expected = RequireStringValue(source, name, path);
        var actual = Sha256(Encoding.UTF8.GetBytes(value));
        if (!string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase))
        {
            throw ContractError(path, $"{name} differs from the character map");
        }
    }

    private static string Sha256(byte[] bytes) =>
        Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static UsageException ContractError(string path, string message) =>
        new($"White line student contract {path}: {message}");
}
