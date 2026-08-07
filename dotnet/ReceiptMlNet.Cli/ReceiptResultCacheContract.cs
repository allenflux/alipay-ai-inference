using System.Text.Json;

/// <summary>
/// Versioned semantics for persisted per-receipt JSON results.
///
/// Model hashes alone are not sufficient cache keys: runtime decoding and
/// delivery-policy behavior can change while every model artifact remains
/// byte-identical.  Persist both values in each result and require an exact
/// match before --skip-existing may reuse it.
/// </summary>
internal static class ReceiptResultCacheContract
{
    public const int SchemaVersion = 1;
    public const string SemanticsVersion = "status-review-only-visible-text-negation-v2";

    public static bool IsCurrent(JsonElement root)
    {
        return root.ValueKind == JsonValueKind.Object
            && root.TryGetProperty("result_schema_version", out var schemaVersion)
            && schemaVersion.ValueKind == JsonValueKind.Number
            && schemaVersion.TryGetInt32(out var parsedSchemaVersion)
            && parsedSchemaVersion == SchemaVersion
            && root.TryGetProperty("result_semantics_version", out var semanticsVersion)
            && semanticsVersion.ValueKind == JsonValueKind.String
            && string.Equals(
                semanticsVersion.GetString(),
                SemanticsVersion,
                StringComparison.Ordinal);
    }
}
