using System.Collections.ObjectModel;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

/// <summary>
/// Verified, Python/Paddle-free PP-OCR delivery package.
///
/// The package is deliberately consumed as a directory rather than three loose
/// ONNX paths: the adjacent contract binds the detector, classifier,
/// recognizer and exact Chinese character table together.  Loading this class
/// verifies every delivered byte before an ONNX session is created.
/// </summary>
internal sealed class PaddleOcrDeliveryBundle
{
    public const string ContractFileName = "paddle_ocr_delivery.contract.json";
    private const int SchemaVersion = 1;
    private const string DeliveryKind = "paddle_ocr_v2_delivery";

    private PaddleOcrDeliveryBundle(
        string directoryPath,
        string contractPath,
        string sourceAuditContractSha256,
        PaddleOcrModelInfo detector,
        PaddleOcrModelInfo recognizer,
        PaddleOcrModelInfo classifier,
        PaddleOcrFileRecord dictionary,
        PaddleOcrSettings settings,
        IReadOnlyList<string> recognitionCharacters,
        IReadOnlyList<string> ctcCharacters,
        long packageSizeBytes)
    {
        BundleDirectory = directoryPath;
        ContractPath = contractPath;
        SourceAuditContractSha256 = sourceAuditContractSha256;
        DetModel = detector;
        RecModel = recognizer;
        ClsModel = classifier;
        Dictionary = dictionary;
        Settings = settings;
        RecognitionCharacters = recognitionCharacters;
        CtcCharacters = ctcCharacters;
        PackageSizeBytes = packageSizeBytes;
    }

    public string BundleDirectory { get; }
    public string DirectoryPath => BundleDirectory;
    public string ContractPath { get; }
    public string SourceAuditContractSha256 { get; }
    public PaddleOcrModelInfo DetModel { get; }
    public PaddleOcrModelInfo RecModel { get; }
    public PaddleOcrModelInfo ClsModel { get; }
    public PaddleOcrModelInfo Detector => DetModel;
    public PaddleOcrModelInfo Recognizer => RecModel;
    public PaddleOcrModelInfo Classifier => ClsModel;
    public PaddleOcrFileRecord Dictionary { get; }

    /// <summary>All dictionary characters in their on-disk order (without CTC blank).</summary>
    public IReadOnlyList<string> RecognitionCharacters { get; }
    public IReadOnlyList<string> Charset => RecognitionCharacters;

    /// <summary>Recognizer output index lookup: index 0 is the CTC blank token.</summary>
    public IReadOnlyList<string> CtcCharacters { get; }

    public PaddleOcrSettings Settings { get; }
    public long PackageSizeBytes { get; }

    public static PaddleOcrDeliveryBundle LoadAndVerify(string deliveryDirectory)
    {
        if (string.IsNullOrWhiteSpace(deliveryDirectory))
        {
            throw new UsageException("--ocr-bundle must be a non-empty delivery directory");
        }

        var directory = Path.GetFullPath(deliveryDirectory);
        if (!Directory.Exists(directory))
        {
            throw new UsageException($"Paddle OCR delivery directory does not exist: {directory}");
        }
        if (Directory.Exists(Path.Combine(directory, "paddle")))
        {
            throw new UsageException(
                "Paddle OCR delivery package must not contain the audit paddle/ source directory; " +
                "run package-delivery and use its output directory instead");
        }

        var contractPath = Path.Combine(directory, ContractFileName);
        if (!File.Exists(contractPath))
        {
            throw new UsageException($"Paddle OCR delivery contract does not exist: {contractPath}");
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(contractPath));
            var contract = document.RootElement;
            RequireObject(contract, "Paddle OCR delivery contract");
            if (ReadRequiredInt(contract, "schema_version", "delivery contract") != SchemaVersion)
            {
                throw new UsageException($"Unsupported Paddle OCR delivery contract schema: {contractPath}");
            }
            if (!string.Equals(ReadRequiredString(contract, "kind", "delivery contract"), DeliveryKind, StringComparison.Ordinal))
            {
                throw new UsageException($"Not a Paddle OCR delivery contract: {contractPath}");
            }

            var sourceAuditHash = ReadRequiredString(contract, "source_audit_contract_sha256", "delivery contract");
            RequireSha256(sourceAuditHash, "delivery contract source_audit_contract_sha256");

            var models = ParseModels(directory, RequireProperty(contract, "models", "delivery contract"));
            var dictionary = ParseFileRecord(
                directory,
                RequireProperty(contract, "dictionary", "delivery contract"),
                "dictionary");
            VerifyFile(dictionary, "dictionary");

            var settings = PaddleOcrSettings.Parse(
                RequireProperty(contract, "effective_paddleocr_args", "delivery contract"),
                models["det"].File.RelativePath,
                models["rec"].File.RelativePath,
                models["cls"].File.RelativePath,
                dictionary.RelativePath);
            var characters = ReadCharset(dictionary.FullPath, settings.UseSpaceChar);
            VerifyRecognizerVocabulary(models["rec"], characters.CtcCharacters);

            var packageSizeBytes = ReadRequiredLong(contract, "package_size_bytes", "delivery contract");
            if (packageSizeBytes < 0)
            {
                throw new UsageException($"Paddle OCR delivery package_size_bytes must not be negative: {contractPath}");
            }
            var calculatedPackageSize = checked(
                models.Values.Sum(model => model.File.SizeBytes) + dictionary.SizeBytes);
            if (packageSizeBytes != calculatedPackageSize)
            {
                throw new UsageException(
                    $"Paddle OCR delivery package_size_bytes differs from the verified files: {contractPath}");
            }

            return new PaddleOcrDeliveryBundle(
                directory,
                contractPath,
                sourceAuditHash.ToLowerInvariant(),
                models["det"],
                models["rec"],
                models["cls"],
                dictionary,
                settings,
                characters.RecognitionCharacters,
                characters.CtcCharacters,
                packageSizeBytes);
        }
        catch (JsonException exception)
        {
            throw new UsageException($"Invalid Paddle OCR delivery contract {contractPath}: {exception.Message}");
        }
        catch (OverflowException exception)
        {
            throw new UsageException($"Invalid numeric value in Paddle OCR delivery contract {contractPath}: {exception.Message}");
        }
    }

    private static IReadOnlyDictionary<string, PaddleOcrModelInfo> ParseModels(string directory, JsonElement modelsElement)
    {
        RequireObject(modelsElement, "delivery contract models");
        var models = new Dictionary<string, PaddleOcrModelInfo>(StringComparer.Ordinal);
        foreach (var property in modelsElement.EnumerateObject())
        {
            if (!models.TryAdd(property.Name, ParseModel(directory, property.Name, property.Value)))
            {
                throw new UsageException($"Paddle OCR delivery contract has duplicate model role: {property.Name}");
            }
        }

        var requiredRoles = new[] { "det", "rec", "cls" };
        if (models.Count != requiredRoles.Length || requiredRoles.Any(role => !models.ContainsKey(role)))
        {
            throw new UsageException("Paddle OCR delivery contract must contain exactly det, rec and cls ONNX models");
        }
        return new ReadOnlyDictionary<string, PaddleOcrModelInfo>(models);
    }

    private static PaddleOcrModelInfo ParseModel(string directory, string role, JsonElement element)
    {
        RequireObject(element, $"{role} model record");
        var file = ParseFileRecord(directory, element, $"{role} model");
        VerifyFile(file, $"{role} model");

        var io = RequireProperty(element, "io", $"{role} model");
        RequireObject(io, $"{role} model io");
        var inputs = ParseTensorContracts(RequireProperty(io, "inputs", $"{role} model io"), $"{role} model input");
        var outputs = ParseTensorContracts(RequireProperty(io, "outputs", $"{role} model io"), $"{role} model output");
        if (inputs.Count != 1)
        {
            throw new UsageException($"Paddle OCR {role} ONNX must expose exactly one input; found {inputs.Count}");
        }
        if (outputs.Count == 0)
        {
            throw new UsageException($"Paddle OCR {role} ONNX must expose at least one output");
        }
        if (inputs[0].Shape.Count != 4)
        {
            throw new UsageException($"Paddle OCR {role} ONNX input must be rank-4 NCHW; found rank {inputs[0].Shape.Count}");
        }
        if (!string.Equals(inputs[0].ElementType, "tensor(float)", StringComparison.Ordinal))
        {
            throw new UsageException($"Paddle OCR {role} ONNX input must be tensor(float); found {inputs[0].ElementType}");
        }

        var dynamic = ParseDynamicShapeRequirement(
            RequireProperty(element, "dynamic_shape_validation", $"{role} model"),
            role,
            inputs[0]);
        return new PaddleOcrModelInfo(role, file, inputs[0], outputs, dynamic);
    }

    private static PaddleOcrFileRecord ParseFileRecord(string directory, JsonElement element, string description)
    {
        RequireObject(element, description);
        var relativePath = ReadRequiredString(element, "path", description);
        var fullPath = ResolveBundleFile(directory, relativePath, description);
        var hash = ReadRequiredString(element, "sha256", description);
        RequireSha256(hash, $"{description} SHA-256");
        var size = ReadRequiredLong(element, "size_bytes", description);
        if (size < 0)
        {
            throw new UsageException($"Paddle OCR {description} size_bytes must not be negative");
        }
        return new PaddleOcrFileRecord(relativePath.Replace('\\', '/'), fullPath, hash.ToLowerInvariant(), size);
    }

    private static IReadOnlyList<PaddleOcrTensorContract> ParseTensorContracts(JsonElement element, string description)
    {
        if (element.ValueKind != JsonValueKind.Array)
        {
            throw new UsageException($"Paddle OCR {description}s must be an array");
        }
        var values = new List<PaddleOcrTensorContract>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var value in element.EnumerateArray())
        {
            RequireObject(value, description);
            var name = ReadRequiredString(value, "name", description);
            if (!names.Add(name))
            {
                throw new UsageException($"Paddle OCR {description}s contain duplicate tensor name: {name}");
            }
            var type = ReadRequiredString(value, "type", description);
            var shapeElement = RequireProperty(value, "shape", description);
            if (shapeElement.ValueKind != JsonValueKind.Array)
            {
                throw new UsageException($"Paddle OCR {description} shape must be an array: {name}");
            }
            var shape = new List<PaddleOcrTensorDimension>();
            foreach (var dimension in shapeElement.EnumerateArray())
            {
                shape.Add(ParseTensorDimension(dimension, description, name));
            }
            values.Add(new PaddleOcrTensorContract(name, shape.AsReadOnly(), type));
        }
        return values.AsReadOnly();
    }

    private static PaddleOcrTensorDimension ParseTensorDimension(JsonElement dimension, string description, string tensorName)
    {
        return dimension.ValueKind switch
        {
            JsonValueKind.Null => new PaddleOcrTensorDimension(null, null),
            JsonValueKind.String => new PaddleOcrTensorDimension(null, dimension.GetString()),
            JsonValueKind.Number when dimension.TryGetInt32(out var value) && value > 0
                => new PaddleOcrTensorDimension(value, null),
            JsonValueKind.Number when dimension.TryGetInt32(out var dynamicValue) && dynamicValue == -1
                => new PaddleOcrTensorDimension(null, null),
            _ => throw new UsageException(
                $"Paddle OCR {description} tensor {tensorName} has invalid shape dimension: {dimension.GetRawText()}"),
        };
    }

    private static PaddleOcrDynamicShapeRequirement ParseDynamicShapeRequirement(
        JsonElement element,
        string role,
        PaddleOcrTensorContract input)
    {
        RequireObject(element, $"{role} dynamic_shape_validation");
        var expectedInputShape = ParseShape(
            RequireProperty(element, "input_shape", $"{role} dynamic_shape_validation"),
            $"{role} dynamic_shape_validation input_shape");
        if (!ShapesEqual(input.Shape, expectedInputShape))
        {
            throw new UsageException($"Paddle OCR {role} dynamic_shape_validation does not match its ONNX input shape");
        }
        var axesElement = RequireProperty(element, "required_dynamic_axes", $"{role} dynamic_shape_validation");
        if (axesElement.ValueKind != JsonValueKind.Array)
        {
            throw new UsageException($"Paddle OCR {role} required_dynamic_axes must be an array");
        }
        var axes = new List<int>();
        foreach (var axisElement in axesElement.EnumerateArray())
        {
            if (!axisElement.TryGetInt32(out var axis) || axis < 0 || axis >= input.Shape.Count || axes.Contains(axis))
            {
                throw new UsageException($"Paddle OCR {role} has an invalid required dynamic axis");
            }
            if (!input.Shape[axis].IsDynamic)
            {
                throw new UsageException($"Paddle OCR {role} lost required dynamic axis {axis}");
            }
            axes.Add(axis);
        }
        return new PaddleOcrDynamicShapeRequirement(expectedInputShape, axes.AsReadOnly());
    }

    private static IReadOnlyList<PaddleOcrTensorDimension> ParseShape(JsonElement element, string description)
    {
        if (element.ValueKind != JsonValueKind.Array)
        {
            throw new UsageException($"Paddle OCR {description} must be an array");
        }
        var shape = new List<PaddleOcrTensorDimension>();
        foreach (var dimension in element.EnumerateArray())
        {
            shape.Add(ParseTensorDimension(dimension, description, "input"));
        }
        return shape.AsReadOnly();
    }

    private static bool ShapesEqual(IReadOnlyList<PaddleOcrTensorDimension> left, IReadOnlyList<PaddleOcrTensorDimension> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }
        for (var index = 0; index < left.Count; index++)
        {
            if (left[index] != right[index])
            {
                return false;
            }
        }
        return true;
    }

    private static (IReadOnlyList<string> RecognitionCharacters, IReadOnlyList<string> CtcCharacters) ReadCharset(
        string charsetPath,
        bool useSpaceCharacter)
    {
        var encoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);
        var characters = File.ReadAllLines(charsetPath, encoding).ToList();
        if (characters.Count == 0)
        {
            throw new UsageException($"Paddle OCR character dictionary is empty: {charsetPath}");
        }
        if (useSpaceCharacter)
        {
            characters.Add(" ");
        }
        var ctcCharacters = new List<string>(characters.Count + 1) { "blank" };
        ctcCharacters.AddRange(characters);
        return (characters.AsReadOnly(), ctcCharacters.AsReadOnly());
    }

    private static void VerifyRecognizerVocabulary(PaddleOcrModelInfo recognizer, IReadOnlyList<string> ctcCharacters)
    {
        var output = recognizer.PrimaryOutput;
        if (output.Shape.Count != 3)
        {
            throw new UsageException(
                $"Paddle OCR recognizer primary output must be rank-3 [batch,time,character]; found rank {output.Shape.Count}");
        }
        var characterDimension = output.Shape[^1].StaticValue;
        if (characterDimension is not null && characterDimension != ctcCharacters.Count)
        {
            throw new UsageException(
                $"Paddle OCR recognizer output character count {characterDimension} does not match dictionary count {ctcCharacters.Count}");
        }
    }

    private static void VerifyFile(PaddleOcrFileRecord record, string description)
    {
        if (!File.Exists(record.FullPath))
        {
            throw new UsageException($"Paddle OCR {description} is missing: {record.FullPath}");
        }
        var actualSize = new FileInfo(record.FullPath).Length;
        if (actualSize != record.SizeBytes)
        {
            throw new UsageException($"Paddle OCR {description} size differs from contract: {record.FullPath}");
        }
        var actualHash = Sha256(record.FullPath);
        if (!string.Equals(actualHash, record.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException($"Paddle OCR {description} SHA-256 differs from contract: {record.FullPath}");
        }
    }

    private static string ResolveBundleFile(string directory, string relativePath, string description)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
        {
            throw new UsageException($"Paddle OCR {description} path is empty");
        }
        var normalized = relativePath.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(normalized))
        {
            throw new UsageException($"Paddle OCR {description} path must be relative to the delivery directory: {relativePath}");
        }
        var fullPath = Path.GetFullPath(Path.Combine(directory, normalized));
        var relativeToRoot = Path.GetRelativePath(directory, fullPath);
        if (relativeToRoot.Equals("..", StringComparison.Ordinal)
            || relativeToRoot.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || Path.IsPathRooted(relativeToRoot))
        {
            throw new UsageException($"Paddle OCR {description} path escapes the delivery directory: {relativePath}");
        }
        return fullPath;
    }

    private static void RequireSha256(string value, string description)
    {
        if (value.Length != 64 || value.Any(character => !Uri.IsHexDigit(character)))
        {
            throw new UsageException($"Paddle OCR {description} must be a 64-character hexadecimal SHA-256");
        }
    }

    private static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        using var algorithm = SHA256.Create();
        return Convert.ToHexString(algorithm.ComputeHash(stream)).ToLowerInvariant();
    }

    internal static JsonElement RequireProperty(JsonElement element, string property, string description)
    {
        if (!element.TryGetProperty(property, out var value))
        {
            throw new UsageException($"Paddle OCR {description} is missing {property}");
        }
        return value;
    }

    internal static void RequireObject(JsonElement element, string description)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new UsageException($"Paddle OCR {description} must be a JSON object");
        }
    }

    internal static string ReadRequiredString(JsonElement element, string property, string description)
    {
        var value = RequireProperty(element, property, description);
        if (value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw new UsageException($"Paddle OCR {description}.{property} must be a non-empty string");
        }
        return value.GetString()!;
    }

    internal static int ReadRequiredInt(JsonElement element, string property, string description)
    {
        var value = RequireProperty(element, property, description);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out var result))
        {
            throw new UsageException($"Paddle OCR {description}.{property} must be an integer");
        }
        return result;
    }

    internal static long ReadRequiredLong(JsonElement element, string property, string description)
    {
        var value = RequireProperty(element, property, description);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt64(out var result))
        {
            throw new UsageException($"Paddle OCR {description}.{property} must be an integer");
        }
        return result;
    }
}

internal sealed record PaddleOcrFileRecord(string RelativePath, string FullPath, string Sha256, long SizeBytes);

internal sealed record PaddleOcrModelInfo(
    string Role,
    PaddleOcrFileRecord File,
    PaddleOcrTensorContract Input,
    IReadOnlyList<PaddleOcrTensorContract> Outputs,
    PaddleOcrDynamicShapeRequirement DynamicShape)
{
    public string FullPath => File.FullPath;
    public string InputName => Input.Name;
    public PaddleOcrTensorContract PrimaryOutput => Outputs[0];
    public string OutputName => PrimaryOutput.Name;
}

internal sealed record PaddleOcrTensorContract(
    string Name,
    IReadOnlyList<PaddleOcrTensorDimension> Shape,
    string ElementType);

internal sealed record PaddleOcrTensorDimension(int? StaticValue, string? Symbol)
{
    public bool IsDynamic => StaticValue is null;
}

internal sealed record PaddleOcrDynamicShapeRequirement(
    IReadOnlyList<PaddleOcrTensorDimension> InputShape,
    IReadOnlyList<int> RequiredDynamicAxes);

/// <summary>
/// The subset of PaddleOCR v2 effective arguments that changes direct ONNX
/// behaviour.  <see cref="Raw"/> retains all audited values for diagnostics.
/// </summary>
internal sealed class PaddleOcrSettings
{
    private PaddleOcrSettings(
        IReadOnlyDictionary<string, JsonElement> raw,
        string language,
        string ocrVersion,
        string detAlgorithm,
        int detLimitSideLength,
        string detLimitType,
        string detBoxType,
        float detDbThreshold,
        float detDbBoxThreshold,
        float detDbUnclipRatio,
        bool useDilation,
        string detDbScoreMode,
        PaddleOcrImageShape recImageShape,
        int recBatchSize,
        string recAlgorithm,
        PaddleOcrImageShape clsImageShape,
        int clsBatchSize,
        float clsThreshold,
        float dropScore,
        bool useSpaceChar)
    {
        Raw = raw;
        Language = language;
        OcrVersion = ocrVersion;
        DetAlgorithm = detAlgorithm;
        DetLimitSideLength = detLimitSideLength;
        DetLimitType = detLimitType;
        DetBoxType = detBoxType;
        DetDbThreshold = detDbThreshold;
        DetDbBoxThreshold = detDbBoxThreshold;
        DetDbUnclipRatio = detDbUnclipRatio;
        UseDilation = useDilation;
        DetDbScoreMode = detDbScoreMode;
        RecImageShape = recImageShape;
        RecBatchSize = recBatchSize;
        RecAlgorithm = recAlgorithm;
        ClsImageShape = clsImageShape;
        ClsBatchSize = clsBatchSize;
        ClsThreshold = clsThreshold;
        DropScore = dropScore;
        UseSpaceChar = useSpaceChar;
    }

    public IReadOnlyDictionary<string, JsonElement> Raw { get; }
    public string Language { get; }
    public string OcrVersion { get; }
    public string DetAlgorithm { get; }
    public int DetLimitSideLength { get; }
    public string DetLimitType { get; }
    public string DetBoxType { get; }
    public float DetDbThreshold { get; }
    public float DetDbBoxThreshold { get; }
    public float DetDbUnclipRatio { get; }
    public bool UseDilation { get; }
    public string DetDbScoreMode { get; }
    public PaddleOcrImageShape RecImageShape { get; }
    public int RecBatchSize { get; }
    public string RecAlgorithm { get; }
    public PaddleOcrImageShape ClsImageShape { get; }
    public int ClsBatchSize { get; }
    public float ClsThreshold { get; }
    public float DropScore { get; }
    public bool UseSpaceChar { get; }

    public static PaddleOcrSettings Parse(
        JsonElement element,
        string detPath,
        string recPath,
        string clsPath,
        string charsetPath)
    {
        PaddleOcrDeliveryBundle.RequireObject(element, "effective_paddleocr_args");
        var raw = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (!raw.TryAdd(property.Name, property.Value.Clone()))
            {
                throw new UsageException($"Paddle OCR effective_paddleocr_args has duplicate key: {property.Name}");
            }
        }

        RequireModelReference(raw, "det_model_dir", detPath);
        RequireModelReference(raw, "rec_model_dir", recPath);
        RequireModelReference(raw, "cls_model_dir", clsPath);
        RequireModelReference(raw, "rec_char_dict_path", charsetPath);
        if (!ReadRequiredBool(raw, "use_onnx") || !ReadRequiredBool(raw, "use_angle_cls"))
        {
            throw new UsageException("Paddle OCR delivery contract must enable use_onnx and use_angle_cls");
        }

        var settings = new PaddleOcrSettings(
            new ReadOnlyDictionary<string, JsonElement>(raw),
            ReadRequiredString(raw, "lang"),
            ReadRequiredString(raw, "ocr_version"),
            ReadRequiredString(raw, "det_algorithm"),
            ReadPositiveInt(raw, "det_limit_side_len"),
            ReadRequiredString(raw, "det_limit_type"),
            ReadRequiredString(raw, "det_box_type"),
            ReadUnitFloat(raw, "det_db_thresh"),
            ReadUnitFloat(raw, "det_db_box_thresh"),
            ReadPositiveFloat(raw, "det_db_unclip_ratio"),
            ReadRequiredBool(raw, "use_dilation"),
            ReadRequiredString(raw, "det_db_score_mode"),
            ReadImageShape(raw, "rec_image_shape"),
            ReadPositiveInt(raw, "rec_batch_num"),
            ReadRequiredString(raw, "rec_algorithm"),
            ReadImageShape(raw, "cls_image_shape"),
            ReadPositiveInt(raw, "cls_batch_num"),
            ReadUnitFloat(raw, "cls_thresh"),
            ReadUnitFloat(raw, "drop_score"),
            ReadRequiredBool(raw, "use_space_char"));

        if (!string.Equals(settings.DetAlgorithm, "DB", StringComparison.Ordinal)
            || !string.Equals(settings.DetBoxType, "quad", StringComparison.Ordinal))
        {
            throw new UsageException(
                $"This .NET adapter supports only PaddleOCR DB quad detection; contract requests {settings.DetAlgorithm}/{settings.DetBoxType}");
        }
        if (settings.DetLimitType is not ("max" or "min" or "resize_long"))
        {
            throw new UsageException($"Unsupported Paddle OCR det_limit_type: {settings.DetLimitType}");
        }
        if (settings.DetDbScoreMode is not ("fast" or "slow"))
        {
            throw new UsageException($"Unsupported Paddle OCR det_db_score_mode: {settings.DetDbScoreMode}");
        }
        if (!string.Equals(settings.RecAlgorithm, "SVTR_LCNet", StringComparison.Ordinal))
        {
            throw new UsageException(
                $"This .NET adapter supports the frozen PP-OCRv4 SVTR_LCNet CTC recognizer; " +
                $"contract requests {settings.RecAlgorithm}");
        }
        var classifierLabels = ReadRequiredStringArray(raw, "label_list");
        if (classifierLabels.Count != 2
            || !string.Equals(classifierLabels[0], "0", StringComparison.Ordinal)
            || !string.Equals(classifierLabels[1], "180", StringComparison.Ordinal))
        {
            throw new UsageException("This .NET adapter requires PaddleOCR classifier label_list [\"0\", \"180\"]");
        }
        return settings;
    }

    private static void RequireModelReference(IReadOnlyDictionary<string, JsonElement> values, string key, string expected)
    {
        var actual = ReadRequiredString(values, key).Replace('\\', '/');
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} does not match the delivered asset path");
        }
    }

    private static string ReadRequiredString(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        if (!values.TryGetValue(key, out var value)
            || value.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a non-empty string");
        }
        return value.GetString()!;
    }

    private static bool ReadRequiredBool(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        if (!values.TryGetValue(key, out var value) || value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a boolean");
        }
        return value.GetBoolean();
    }

    private static IReadOnlyList<string> ReadRequiredStringArray(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        if (!values.TryGetValue(key, out var value) || value.ValueKind != JsonValueKind.Array)
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a string array");
        }
        var result = new List<string>();
        foreach (var item in value.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
            {
                throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a string array");
            }
            result.Add(item.GetString()!);
        }
        return result.AsReadOnly();
    }

    private static int ReadPositiveInt(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        if (!values.TryGetValue(key, out var value) || value.ValueKind != JsonValueKind.Number)
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a positive integer");
        }
        if (value.TryGetInt32(out var result) && result > 0)
        {
            return result;
        }
        // PaddleOCR declares det_limit_side_len as a float. Its default is
        // consequently serialised by the audit snapshot as an integer-like
        // value such as 960.0 rather than necessarily JSON integer 960.
        if (value.TryGetDouble(out var decimalValue)
            && double.IsFinite(decimalValue)
            && decimalValue > 0.0
            && decimalValue <= int.MaxValue
            && Math.Truncate(decimalValue) == decimalValue)
        {
            return (int)decimalValue;
        }
        throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a positive integer");
    }

    private static float ReadUnitFloat(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        var result = ReadFiniteFloat(values, key);
        if (result is < 0.0f or > 1.0f)
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be between 0 and 1");
        }
        return result;
    }

    private static float ReadPositiveFloat(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        var result = ReadFiniteFloat(values, key);
        if (result <= 0.0f)
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be positive");
        }
        return result;
    }

    private static float ReadFiniteFloat(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        if (!values.TryGetValue(key, out var value)
            || value.ValueKind != JsonValueKind.Number
            || !value.TryGetSingle(out var result)
            || !float.IsFinite(result))
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a finite number");
        }
        return result;
    }

    private static PaddleOcrImageShape ReadImageShape(IReadOnlyDictionary<string, JsonElement> values, string key)
    {
        if (!values.TryGetValue(key, out var value))
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} is missing");
        }
        var dimensions = new List<int>();
        if (value.ValueKind == JsonValueKind.String)
        {
            foreach (var part in value.GetString()!.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                if (!int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out var dimension))
                {
                    throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} contains a non-integer dimension");
                }
                dimensions.Add(dimension);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in value.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out var dimension))
                {
                    throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} contains a non-integer dimension");
                }
                dimensions.Add(dimension);
            }
        }
        else
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must be a comma-separated string or integer array");
        }

        if (dimensions.Count != 3 || dimensions.Any(value => value <= 0))
        {
            throw new UsageException($"Paddle OCR effective_paddleocr_args.{key} must contain three positive C,H,W dimensions");
        }
        return new PaddleOcrImageShape(dimensions[0], dimensions[1], dimensions[2]);
    }
}

internal sealed record PaddleOcrImageShape(int Channels, int Height, int Width);
