using System.Text.Json;
using Headroom.GeneratedPilot;

try
{
    JsonSerializer.Serialize(new RetrieveRequest {
        Hash = "ok",
        AdditionalProperties = new() { ["hash"] = JsonSerializer.SerializeToElement("other") }
    });
    throw new Exception("Extension property shadowed request hash");
}
catch (JsonException) { }

const string valid = """
    {"hash":"h","original_content":"x","original_tokens":1,"original_item_count":1,"compressed_item_count":1,"tool_name":null,"retrieval_count":1,"future_extension":{"snake_case":"unchanged"}}
    """;
var parsed = JsonSerializer.Deserialize<RetrieveResponse>(valid)
    ?? throw new Exception("Explicit-null response decoded to null");
if (parsed.ToolName is not null)
    throw new Exception("Explicit null was not preserved");
if (parsed.AdditionalProperties["future_extension"].GetProperty("snake_case").GetString() != "unchanged")
    throw new Exception("Unknown response property was not preserved");

const string missing = """
    {"hash":"h","original_content":"x","original_tokens":1,"original_item_count":1,"compressed_item_count":1,"retrieval_count":1}
    """;
try
{
    JsonSerializer.Deserialize<RetrieveResponse>(missing);
    throw new Exception("Missing required-nullable tool_name was accepted");
}
catch (JsonException)
{
    // Expected: required nullable means the property must be present, though its value may be null.
}

Console.WriteLine("PASS: .NET required-nullable and unknown-field model checks");

var wireUrl = Environment.GetEnvironmentVariable("HEADROOM_WIRE_TEST_URL");
if (wireUrl is not null)
{
    using var client = new Client(wireUrl);
    var response = await client.RetrieveAsync(new RetrieveRequest
    {
        Hash = "ok",
        AdditionalProperties = new() { ["extra_request"] = JsonSerializer.SerializeToElement(new { snake_case = false }) },
    });
    if (response.ToolName is not null || response.OriginalContent != "{\"snake_case\":\"世界\"}")
        throw new Exception("POST response values were not preserved");
    if (response.AdditionalProperties["future_extension"].GetProperty("snake_case").GetString() != "unchanged")
        throw new Exception("POST unknown response field was not preserved");
    try
    {
        await client.RetrieveAsync(new RetrieveRequest { Hash = null! });
        throw new Exception("Null non-nullable request field was accepted");
    }
    catch (ProtocolException) { }

    const string key = "a/世界 ?#+'!*()";
    var byHash = await client.RetrieveGetAsync(key);
    if (byHash.Hash != key) throw new Exception("Encoded GET path did not round-trip");

    try
    {
        await client.RetrieveAsync(new RetrieveRequest { Hash = "missing" });
        throw new Exception("HTTP 404 was accepted");
    }
    catch (APIException error) when ((int)error.StatusCode == 404)
    {
        if (!System.Text.Encoding.UTF8.GetString(error.Body).Contains("Entry missing", StringComparison.Ordinal))
            throw new Exception("HTTP error body was not retained");
        if (error.Message.Contains("Entry missing", StringComparison.Ordinal))
            throw new Exception("HTTP error display leaked its body");
    }

    foreach (var bad in new[] { "missing_field", "wrong_type", "null_nonnullable", "notjson", "valid_nonjson" })
    {
        try
        {
            await client.RetrieveAsync(new RetrieveRequest { Hash = bad });
            throw new Exception($"Malformed response {bad} was accepted");
        }
        catch (ProtocolException) { }
    }

    try
    {
        await client.RetrieveAsync(new RetrieveRequest { Hash = "retry_probe_dotnet" });
        throw new Exception("Closed connection was retried or accepted");
    }
    catch (HttpRequestException) { }

    var wide = await client.RetrieveAsync(new RetrieveRequest { Hash = "unsafe_int" });
    if (wide.OriginalTokens != 9007199254740993L) throw new Exception("Wide integer changed");

    try
    {
        await client.RetrieveAsync(new RetrieveRequest { Hash = "redirect" });
        throw new Exception("Redirect was followed or accepted");
    }
    catch (APIException error) when ((int)error.StatusCode == 302) { }

    using var bounded = new Client(wireUrl, new ClientOptions(TimeSpan.FromSeconds(30), 16));
    try
    {
        await bounded.RetrieveAsync(new RetrieveRequest { Hash = "ok" });
        throw new Exception("Oversized response was accepted");
    }
    catch (ProtocolException) { }

    try
    {
        await client.RetrieveGetAsync("..");
        throw new Exception("Dot path segment was accepted");
    }
    catch (ArgumentException) { }

    foreach (var badBase in new[] { "file:///tmp", "http://user:pass@localhost", "http://localhost/?secret=1" })
    {
        try
        {
            using var invalid = new Client(badBase);
            throw new Exception($"Invalid base URL was accepted: {badBase}");
        }
        catch (ArgumentException) { }
    }

    using var timed = new Client(wireUrl, new ClientOptions(TimeSpan.FromMilliseconds(50), 1024));
    try
    {
        await timed.RetrieveAsync(new RetrieveRequest { Hash = "slow" });
        throw new Exception("Timed out response was accepted");
    }
    catch (OperationCanceledException) { }

    using var cancellation = new CancellationTokenSource();
    cancellation.CancelAfter(TimeSpan.FromMilliseconds(25));
    try
    {
        await client.RetrieveAsync(new RetrieveRequest { Hash = "slow" }, cancellation.Token);
        throw new Exception("Caller cancellation was ignored");
    }
    catch (OperationCanceledException) { }

    Console.WriteLine("PASS: .NET loopback HTTP conformance checks");
}
