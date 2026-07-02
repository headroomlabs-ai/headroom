"""LiteLLM-based backend for Headroom.

Uses LiteLLM to support 100+ providers with minimal code:
- AWS Bedrock: model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
- Azure OpenAI: model="azure/gpt-4"
- Google Vertex: model="vertex_ai/claude-3-5-sonnet"
- OpenRouter: model="openrouter/anthropic/claude-3.5-sonnet"
- And many more...

LiteLLM handles all the auth and format translation internally.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .base import Backend, BackendResponse, StreamEvent

logger = logging.getLogger(__name__)

# litellm calls `dotenv.load_dotenv()` during its own import, which loads
# the project `.env` into `os.environ`. We don't want that side effect —
# importing a backend module should not silently leak API keys into the
# process. Snapshot `os.environ` around the import and undo any keys
# litellm added. Same pattern as `headroom/pricing/litellm_pricing.py`.
try:
    import os as _os

    _env_snapshot = set(_os.environ)
    import litellm
    from litellm import acompletion

    for _leaked_key in set(_os.environ) - _env_snapshot:
        del _os.environ[_leaked_key]
    del _env_snapshot, _os

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore
    acompletion = None  # type: ignore


# =============================================================================
# Provider Registry - Add new providers here!
# =============================================================================


@dataclass
class ProviderConfig:
    """Configuration for a LiteLLM provider."""

    name: str  # Provider identifier (e.g., "bedrock", "openrouter")
    display_name: str  # Human-readable name (e.g., "AWS Bedrock", "OpenRouter")
    model_map: dict[str, str] = field(default_factory=dict)  # Anthropic -> provider model map
    pass_through: bool = False  # If True, prepend provider/ to any model
    uses_region: bool = True  # Whether region is relevant for this provider
    env_vars: list[str] = field(default_factory=list)  # Required env vars
    model_format_hint: str = ""  # Hint for model naming (shown in help)


# Cache for dynamically fetched inference profiles
_bedrock_profiles_cache: dict[str, dict[str, str]] = {}  # region -> model_map

# Region prefix used in cross-region Bedrock inference profile IDs.
# EU regions use "eu.", AP regions use "apac.", US (and everything else) use "us.".
_BEDROCK_REGION_PREFIXES: dict[str, str] = {
    "eu": "eu",
    "ap": "apac",
}


def _bedrock_region_prefix(region: str) -> str:
    """Return the inference-profile region prefix for an AWS region.

    AWS Bedrock cross-region inference profiles are prefixed with a
    geographic tag: ``us.``, ``eu.``, or ``apac.``.  This helper maps
    an AWS region name (e.g. ``eu-west-1``) to the correct prefix.

    >>> _bedrock_region_prefix("us-east-1")
    'us'
    >>> _bedrock_region_prefix("eu-central-1")
    'eu'
    >>> _bedrock_region_prefix("ap-southeast-1")
    'apac'
    """
    for key, prefix in _BEDROCK_REGION_PREFIXES.items():
        if region.startswith(key):
            return prefix
    return "us"


def _build_bedrock_fallback_map(region: str) -> dict[str, str]:
    """Build a static Bedrock model map using the region prefix.

    When ``_fetch_bedrock_inference_profiles`` cannot reach the AWS API
    (wrong credentials, network error, permissions, etc.) we fall back
    to this map so that the proxy can still route requests.  The map
    covers all currently GA Claude models on Bedrock.
    """
    prefix = _bedrock_region_prefix(region)

    # Base model IDs without region prefix
    _CLAUDE_MODELS = [
        # Claude 4.6
        ("claude-opus-4-6", "anthropic.claude-opus-4-6-v1"),
        ("claude-sonnet-4-6", "anthropic.claude-sonnet-4-6"),
        # Claude 4.5
        ("claude-sonnet-4-5-20250929", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("claude-opus-4-5-20251101", "anthropic.claude-opus-4-5-20251101-v1:0"),
        # Claude 4.1
        ("claude-opus-4-1-20250805", "anthropic.claude-opus-4-1-20250805-v1:0"),
        # Claude 4
        ("claude-sonnet-4-20250514", "anthropic.claude-sonnet-4-20250514-v1:0"),
        ("claude-opus-4-20250514", "anthropic.claude-opus-4-20250514-v1:0"),
        # Claude 3.7
        ("claude-3-7-sonnet-20250219", "anthropic.claude-3-7-sonnet-20250219-v1:0"),
        # Claude 3.5
        ("claude-3-5-sonnet-20241022", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("claude-3-5-sonnet-20240620", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        ("claude-3-5-haiku-20241022", "anthropic.claude-3-5-haiku-20241022-v1:0"),
        # Claude 3
        ("claude-3-opus-20240229", "anthropic.claude-3-opus-20240229-v1:0"),
        ("claude-3-sonnet-20240229", "anthropic.claude-3-sonnet-20240229-v1:0"),
        ("claude-3-haiku-20240307", "anthropic.claude-3-haiku-20240307-v1:0"),
        # Haiku 4.5
        ("claude-haiku-4-5-20251001", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    ]

    return {name: f"bedrock/{prefix}.{model_id}" for name, model_id in _CLAUDE_MODELS}


def _fetch_bedrock_inference_profiles(region: str | None) -> dict[str, str]:
    """Fetch available Bedrock inference profiles from AWS API.

    Uses boto3 list_inference_profiles() to get all available profiles
    for the given region, then builds a model map.

    If the API call fails (wrong credentials, network error, permission
    denied, etc.) the function logs a warning and returns a static
    fallback map so the proxy can still start.

    Args:
        region: AWS region (e.g., "us-east-1", "eu-central-1")

    Returns:
        Model map: anthropic_model_name -> bedrock inference profile ID
    """
    region = region or "us-east-1"

    # Check cache first
    if region in _bedrock_profiles_cache:
        return _bedrock_profiles_cache[region]

    model_map: dict[str, str] = {}

    try:
        import boto3
    except ImportError:
        logger.warning(
            "boto3 is not installed — using static Bedrock model map. "
            "Install boto3 for dynamic model discovery: pip install boto3"
        )
        model_map = _build_bedrock_fallback_map(region)
        _bedrock_profiles_cache[region] = model_map
        return model_map

    try:
        bedrock_client = boto3.client("bedrock", region_name=region)
        response = bedrock_client.list_inference_profiles(typeEquals="SYSTEM_DEFINED")

        for profile in response.get("inferenceProfileSummaries", []):
            profile_id = profile.get("inferenceProfileId", "")

            # Only process Anthropic Claude profiles
            if "anthropic" not in profile_id.lower():
                continue

            # Extract the standard model name from the profile ID
            # e.g., "us.anthropic.claude-sonnet-4-20250514-v1:0" -> "claude-sonnet-4-20250514"
            normalized = _normalize_bedrock_profile_id(profile_id)
            if normalized:
                model_map[normalized] = f"bedrock/{profile_id}"

        # Handle pagination if needed
        while response.get("nextToken"):
            response = bedrock_client.list_inference_profiles(
                typeEquals="SYSTEM_DEFINED", nextToken=response["nextToken"]
            )
            for profile in response.get("inferenceProfileSummaries", []):
                profile_id = profile.get("inferenceProfileId", "")
                if "anthropic" not in profile_id.lower():
                    continue
                normalized = _normalize_bedrock_profile_id(profile_id)
                if normalized:
                    model_map[normalized] = f"bedrock/{profile_id}"

        logger.info(f"Fetched {len(model_map)} Bedrock inference profiles for region {region}")
    except Exception as e:
        logger.warning(
            f"Failed to fetch Bedrock inference profiles for region {region}: {e}. "
            "Using static fallback model map."
        )
        model_map = _build_bedrock_fallback_map(region)

    # Cache the result
    _bedrock_profiles_cache[region] = model_map
    return model_map


def _normalize_bedrock_profile_id(profile_id: str) -> str | None:
    """Extract standard Anthropic model name from Bedrock profile ID.

    Args:
        profile_id: e.g., "us.anthropic.claude-sonnet-4-20250514-v1:0"
                    or "anthropic.claude-sonnet-4-20250514-v1:0"
                    or "claude-sonnet-4-20250514"

    Returns:
        Normalized name like "claude-sonnet-4-20250514", or None if not parseable
    """
    import re

    # Strip "bedrock/" prefix if present
    if profile_id.startswith("bedrock/"):
        profile_id = profile_id[8:]

    # Strip region prefix (us., eu., apac.)
    for prefix in ["us.", "eu.", "apac."]:
        if profile_id.startswith(prefix):
            profile_id = profile_id[len(prefix) :]
            break

    # Strip "anthropic." prefix
    if profile_id.startswith("anthropic."):
        profile_id = profile_id[10:]

    # Must be a Claude model
    if not profile_id.startswith("claude"):
        return None

    # Strip version suffix (-v1:0, -v2:0, etc.)
    normalized = re.sub(r"-v\d+:\d+$", "", profile_id)
    return normalized if normalized else None


# Legacy static map - kept for non-Bedrock providers
_BEDROCK_MODEL_MAP: dict[str, str] = {}

_VERTEX_MODEL_MAP = {
    # Claude 4.6 (latest, no date suffix)
    "claude-opus-4-6": "vertex_ai/claude-opus-4-6",
    "claude-sonnet-4-6": "vertex_ai/claude-sonnet-4-6",
    # Claude 4.5
    "claude-sonnet-4-5-20250929": "vertex_ai/claude-sonnet-4-5@20250929",
    "claude-opus-4-5-20251101": "vertex_ai/claude-opus-4-5@20251101",
    # Claude 4.1
    "claude-opus-4-1-20250805": "vertex_ai/claude-opus-4-1@20250805",
    # Claude 4
    "claude-sonnet-4-20250514": "vertex_ai/claude-sonnet-4@20250514",
    "claude-opus-4-20250514": "vertex_ai/claude-opus-4@20250514",
    # Claude 3.7
    "claude-3-7-sonnet-20250219": "vertex_ai/claude-3-7-sonnet@20250219",
    # Claude 3.5
    "claude-3-5-sonnet-20241022": "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet-20240620": "vertex_ai/claude-3-5-sonnet@20240620",
    "claude-3-5-haiku-20241022": "vertex_ai/claude-3-5-haiku@20241022",
    # Claude 3 (haiku 3 deprecated, others retired)
    "claude-3-opus-20240229": "vertex_ai/claude-3-opus@20240229",
    "claude-3-sonnet-20240229": "vertex_ai/claude-3-sonnet@20240229",
    "claude-3-haiku-20240307": "vertex_ai/claude-3-haiku@20240307",
    # Haiku 4.5
    "claude-haiku-4-5-20251001": "vertex_ai/claude-haiku-4-5@20251001",
}


# Provider Registry - to add a new provider, just add an entry here!
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "bedrock": ProviderConfig(
        name="bedrock",
        display_name="AWS Bedrock",
        model_map=_BEDROCK_MODEL_MAP,
        uses_region=True,
        env_vars=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    ),
    "vertex_ai": ProviderConfig(
        name="vertex_ai",
        display_name="Google Vertex AI",
        model_map=_VERTEX_MODEL_MAP,
        uses_region=True,
        env_vars=["GOOGLE_APPLICATION_CREDENTIALS"],
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        model_map={},  # No static map - pass through
        pass_through=True,
        uses_region=False,
        env_vars=["OPENROUTER_API_KEY"],
        model_format_hint="anthropic/claude-3.5-sonnet, openai/gpt-4o, etc.",
    ),
    "azure": ProviderConfig(
        name="azure",
        display_name="Azure OpenAI",
        model_map={},
        uses_region=True,
        env_vars=["AZURE_API_KEY", "AZURE_API_BASE"],
    ),
    "databricks": ProviderConfig(
        name="databricks",
        display_name="Databricks",
        model_map={},  # Pass through - Databricks uses custom model names
        pass_through=True,
        uses_region=False,
        env_vars=["DATABRICKS_API_KEY", "DATABRICKS_API_BASE"],
        model_format_hint="databricks-meta-llama-3-1-70b-instruct, databricks-dbrx-instruct, etc.",
    ),
}


def get_provider_config(provider: str) -> ProviderConfig:
    """Get provider config, with fallback for unknown providers."""
    if provider in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[provider]
    # Fallback for unknown providers - basic pass-through
    return ProviderConfig(
        name=provider,
        display_name=provider.upper(),
        model_map={},
        pass_through=True,
    )


def _convert_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic tool format to OpenAI function format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    func: dict[str, Any] = {"name": tool.get("name", "")}
    if "description" in tool:
        func["description"] = tool["description"]
    if "input_schema" in tool:
        func["parameters"] = tool["input_schema"]
    return {"type": "function", "function": func}


def _convert_tool_choice(choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
    OpenAI:    "auto", "required", {"type": "function", "function": {"name": "..."}}
    """
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        choice_type = choice.get("type", "auto")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {"type": "function", "function": {"name": choice.get("name", "")}}
    return "auto"


def _parse_tool_arguments(arguments: Any) -> Any:
    """Parse tool call arguments from string to dict.

    LiteLLM/OpenAI returns arguments as a JSON string,
    but Anthropic expects input as a parsed dict.
    """
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return arguments
    return arguments


class LiteLLMBackend(Backend):
    """Backend using LiteLLM for multi-provider support.

    Supports any provider LiteLLM supports:
    - bedrock: AWS Bedrock (uses AWS credentials)
    - vertex_ai: Google Vertex AI (uses GCP credentials)
    - openrouter: OpenRouter (400+ models via single API)
    - azure: Azure OpenAI (uses Azure credentials)
    - And 100+ more...

    To add a new provider, just add an entry to PROVIDER_REGISTRY above.
    """

    def __init__(
        self,
        provider: str = "bedrock",
        region: str | None = None,
        **kwargs: Any,
    ):
        """Initialize LiteLLM backend.

        Args:
            provider: LiteLLM provider prefix (bedrock, vertex_ai, openrouter, etc.)
            region: Cloud region (provider-specific)
            **kwargs: Additional provider-specific config
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LiteLLMBackend. Install with: pip install litellm"
            )

        self.provider = provider
        self.region = region
        self.kwargs = kwargs

        # Get provider config from registry
        self._config = get_provider_config(provider)

        # For Bedrock, fetch model map dynamically from AWS API
        if provider == "bedrock":
            # litellm takes the botocore-backed `_auth_with_aws_session_token`
            # path as soon as temporary credentials (AWS_SESSION_TOKEN) are
            # present. botocore is an optional dependency (the `bedrock`
            # extra); when it is absent — as in the slim default Docker image —
            # the failure only surfaces at request time as a misleading
            # `authentication_error: No module named 'botocore'` (#1551). Fail
            # fast at startup with an actionable message instead.
            if os.environ.get("AWS_SESSION_TOKEN") and importlib.util.find_spec("botocore") is None:
                raise ImportError(
                    "Bedrock with temporary credentials (AWS_SESSION_TOKEN) requires "
                    "botocore, which is not installed. Install the bedrock extra: "
                    "pip install 'headroom-ai[bedrock]' (or pip install botocore)."
                )
            self._model_map = _fetch_bedrock_inference_profiles(region)
            litellm.set_verbose = False  # Reduce noise
        else:
            self._model_map = self._config.model_map

        logger.info(f"LiteLLM backend initialized (provider={provider}, region={region})")

    @property
    def name(self) -> str:
        return f"litellm-{self.provider}"

    def map_model_id(self, anthropic_model: str) -> str:
        """Map Anthropic model ID to LiteLLM model string.

        Handles various input formats:
        - "claude-sonnet-4-20250514" (standard Anthropic)
        - "anthropic.claude-sonnet-4-20250514-v1:0" (Bedrock without region)
        - "us.anthropic.claude-sonnet-4-20250514-v1:0" (Bedrock with region)
        - "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0" (LiteLLM format)
        """
        # Check direct mapping first
        if anthropic_model in self._model_map:
            return self._model_map[anthropic_model]

        # For Bedrock, try to normalize various input formats
        if self.provider == "bedrock":
            normalized = _normalize_bedrock_profile_id(anthropic_model)
            if normalized and normalized in self._model_map:
                return self._model_map[normalized]

            # Bedrock fallback: construct a valid region-prefixed model ID.
            # Without this, bare model names like "claude-sonnet-4-20250514"
            # would become "bedrock/claude-sonnet-4-20250514" which is not a
            # valid Bedrock model identifier.
            if "/" not in anthropic_model and anthropic_model.startswith("claude"):
                region_prefix = _bedrock_region_prefix(self.region or "us-east-1")
                return f"bedrock/{region_prefix}.anthropic.{anthropic_model}-v1:0"

        # Pass-through providers: prepend provider prefix
        if self._config.pass_through:
            # If already has provider prefix, use as-is
            if anthropic_model.startswith(f"{self.provider}/"):
                return anthropic_model
            # Otherwise prepend provider/
            return f"{self.provider}/{anthropic_model}"

        # If already has provider prefix, use as-is
        if "/" in anthropic_model:
            return anthropic_model

        # Fallback: construct provider/model format
        return f"{self.provider}/{anthropic_model}"

    def supports_model(self, model: str) -> bool:
        """Check if model is supported."""
        # Pass-through providers accept any model
        if self._config.pass_through:
            return True
        return "claude" in model.lower() or model in self._model_map

    def _convert_messages_for_litellm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic message format to LiteLLM/OpenAI format.

        Anthropic and OpenAI have different representations for tool calls:
        - Anthropic: assistant content blocks with type=tool_use, user content blocks with type=tool_result
        - OpenAI: assistant message with tool_calls field, separate role=tool messages

        This method converts Anthropic-style messages to OpenAI-style so LiteLLM
        can send them to any provider.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle string content directly
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            # Handle content blocks (Anthropic style)
            if isinstance(content, list):
                # Separate blocks by type
                text_parts = []
                tool_use_blocks = []
                tool_result_blocks = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        tool_use_blocks.append(block)
                    elif block_type == "tool_result":
                        tool_result_blocks.append(block)

                # tool_result blocks → OpenAI "tool" role messages
                if tool_result_blocks:
                    # Do NOT insert a separate user text message here — Bedrock
                    # requires tool role messages to appear immediately after the
                    # assistant tool_calls message with no intervening messages.
                    # Any text alongside tool_result is discarded (Claude Code
                    # doesn't send text with tool_result blocks in practice).
                    for tr in tool_result_blocks:
                        tr_content = tr.get("content", "")
                        if isinstance(tr_content, list):
                            tr_content = "\n".join(
                                b.get("text", "") for b in tr_content if b.get("type") == "text"
                            )
                        converted.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr["tool_use_id"],
                                "content": str(tr_content),
                            }
                        )
                    continue

                # tool_use blocks → OpenAI assistant message with tool_calls
                if tool_use_blocks:
                    assistant_msg: dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        assistant_msg["content"] = "\n".join(text_parts)
                    else:
                        assistant_msg["content"] = None
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tu["id"],
                            "type": "function",
                            "function": {
                                "name": tu["name"],
                                "arguments": json.dumps(tu.get("input", {})),
                            },
                        }
                        for tu in tool_use_blocks
                    ]
                    converted.append(assistant_msg)
                    continue

                # Simple text only
                if text_parts:
                    converted.append({"role": role, "content": "\n".join(text_parts)})
                else:
                    converted.append({"role": role, "content": ""})

        return converted

    def _to_anthropic_response(
        self,
        litellm_response: Any,
        original_model: str,
    ) -> dict[str, Any]:
        """Convert LiteLLM/OpenAI response to Anthropic format."""
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Extract content from OpenAI format
        choice = litellm_response.choices[0]
        message = choice.message

        # Build Anthropic content blocks
        content = []
        if message.content:
            content.append({"type": "text", "text": message.content})

        # Handle tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": _parse_tool_arguments(tc.function.arguments),
                    }
                )

        # Map stop reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }
        stop_reason = stop_reason_map.get(choice.finish_reason, "end_turn")

        # Build usage
        usage = {
            "input_tokens": getattr(litellm_response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(litellm_response.usage, "completion_tokens", 0),
        }

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    async def send_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send message via LiteLLM."""
        original_model = body.get("model", "claude-3-5-sonnet-20241022")
        litellm_model = self.map_model_id(original_model)

        try:
            # Convert messages
            messages = self._convert_messages_for_litellm(body.get("messages", []))

            # Build kwargs for litellm
            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": messages,
            }

            # Optional parameters
            if "max_tokens" in body:
                kwargs["max_tokens"] = body["max_tokens"]
            if "temperature" in body:
                kwargs["temperature"] = body["temperature"]
            if "top_p" in body:
                kwargs["top_p"] = body["top_p"]
            if "stop_sequences" in body:
                kwargs["stop"] = body["stop_sequences"]

            # Tools (convert Anthropic format to OpenAI format)
            if "tools" in body:
                kwargs["tools"] = [_convert_anthropic_tool(t) for t in body["tools"]]
            if "tool_choice" in body:
                kwargs["tool_choice"] = _convert_tool_choice(body["tool_choice"])

            # System prompt (Anthropic puts it in body, OpenAI in messages)
            if "system" in body:
                system = body["system"]
                if isinstance(system, str):
                    kwargs["messages"].insert(0, {"role": "system", "content": system})
                elif isinstance(system, list):
                    # Anthropic list format
                    system_text = " ".join(
                        s.get("text", "") if isinstance(s, dict) else str(s) for s in system
                    )
                    kwargs["messages"].insert(0, {"role": "system", "content": system_text})

            # Provider-specific region config
            if self.region:
                if self.provider == "bedrock":
                    kwargs["aws_region_name"] = self.region
                elif self.provider in ("vertex_ai", "vertex_ai_beta"):
                    kwargs["vertex_location"] = self.region

            # Forward API key from request headers if present.
            # Skip for Bedrock/Vertex: they use env-based auth (AWS SigV4 / Google ADC).
            # Forwarding x-api-key (e.g. sk-ant-dummy) would override their credentials.
            _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
            if self.provider not in _env_auth_providers:
                auth_header = headers.get("authorization", headers.get("Authorization", ""))
                if auth_header.startswith("Bearer "):
                    kwargs["api_key"] = auth_header[7:]
                elif headers.get("x-api-key"):
                    kwargs["api_key"] = headers["x-api-key"]

            logger.debug(f"LiteLLM request: model={litellm_model}")

            # Make the call
            response = await acompletion(**kwargs)

            # Convert to Anthropic format
            anthropic_response = self._to_anthropic_response(response, original_model)

            return BackendResponse(
                body=anthropic_response,
                status_code=200,
                headers={"content-type": "application/json"},
            )

        except Exception as e:
            logger.error(f"LiteLLM error: {e}")

            # Map to Anthropic error format
            error_type = "api_error"
            status_code = 500

            error_str = str(e).lower()
            if "authentication" in error_str or "credentials" in error_str:
                error_type = "authentication_error"
                status_code = 401
            elif "rate" in error_str or "limit" in error_str:
                error_type = "rate_limit_error"
                status_code = 429
            elif "not found" in error_str:
                error_type = "not_found_error"
                status_code = 404

            return BackendResponse(
                body={
                    "type": "error",
                    "error": {"type": error_type, "message": str(e)},
                },
                status_code=status_code,
                error=str(e),
            )

    async def stream_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[StreamEvent]:
        """Stream message via LiteLLM.

        Translates OpenAI streaming chunks into Anthropic SSE events.
        Handles both text content and tool_calls dynamically — block types
        are emitted based on what LiteLLM actually returns, not hardcoded.
        """
        original_model = body.get("model", "claude-3-5-sonnet-20241022")
        litellm_model = self.map_model_id(original_model)

        try:
            messages = self._convert_messages_for_litellm(body.get("messages", []))

            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": messages,
                "stream": True,
            }

            if "max_tokens" in body:
                kwargs["max_tokens"] = body["max_tokens"]
            if "temperature" in body:
                kwargs["temperature"] = body["temperature"]
            if "top_p" in body:
                kwargs["top_p"] = body["top_p"]
            if "stop_sequences" in body:
                kwargs["stop"] = body["stop_sequences"]
            if "tools" in body:
                kwargs["tools"] = [_convert_anthropic_tool(t) for t in body["tools"]]
            if "tool_choice" in body:
                kwargs["tool_choice"] = _convert_tool_choice(body["tool_choice"])
            if "system" in body:
                system = body["system"]
                if isinstance(system, str):
                    kwargs["messages"].insert(0, {"role": "system", "content": system})
                elif isinstance(system, list):
                    system_text = " ".join(
                        s.get("text", "") if isinstance(s, dict) else str(s) for s in system
                    )
                    kwargs["messages"].insert(0, {"role": "system", "content": system_text})

            # Provider-specific region config
            if self.region:
                if self.provider == "bedrock":
                    kwargs["aws_region_name"] = self.region
                elif self.provider in ("vertex_ai", "vertex_ai_beta"):
                    kwargs["vertex_location"] = self.region

            # Forward API key from request headers if present.
            # Skip for Bedrock/Vertex: they use env-based auth (AWS SigV4 / Google ADC).
            # Forwarding x-api-key (e.g. sk-ant-dummy) would override their credentials.
            _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
            if self.provider not in _env_auth_providers:
                auth_header = headers.get("authorization", headers.get("Authorization", ""))
                if auth_header.startswith("Bearer "):
                    kwargs["api_key"] = auth_header[7:]
                elif headers.get("x-api-key"):
                    kwargs["api_key"] = headers["x-api-key"]

            msg_id = f"msg_{uuid.uuid4().hex[:24]}"

            # Request LiteLLM to include usage in the final streaming chunk so
            # we can surface real input_tokens in message_start (issue #1132).
            # Without include_usage=True, the final chunk carries no usage and
            # message_start always emits input_tokens=0, breaking OTel metrics.
            if "stream_options" not in kwargs:
                kwargs["stream_options"] = {"include_usage": True}
            elif isinstance(kwargs.get("stream_options"), dict):
                kwargs["stream_options"].setdefault("include_usage", True)

            # Buffer all chunks so we can extract real token counts from the
            # final usage chunk before emitting message_start.
            response = await acompletion(**kwargs)
            buffered_chunks: list[Any] = []
            input_tokens = 0
            output_tokens_from_usage = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

            async for chunk in response:
                buffered_chunks.append(chunk)
                # LiteLLM emits usage on the final chunk when include_usage=True.
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    input_tokens = int(getattr(chunk.usage, "prompt_tokens", 0) or 0)
                    output_tokens_from_usage = int(
                        getattr(chunk.usage, "completion_tokens", 0) or 0
                    )
                    cache_read_input_tokens = int(
                        getattr(chunk.usage, "cache_read_input_tokens", 0) or 0
                    )
                    cache_creation_input_tokens = int(
                        getattr(chunk.usage, "cache_creation_input_tokens", 0) or 0
                    )

            # Build usage dict for message_start — include cache fields only
            # when the backend actually reported them (avoids confusing 0 with
            # "confirmed no cache" vs "backend didn't report").
            msg_start_usage: dict[str, int] = {
                "input_tokens": input_tokens,
                "output_tokens": 0,
            }
            if cache_read_input_tokens:
                msg_start_usage["cache_read_input_tokens"] = cache_read_input_tokens
            if cache_creation_input_tokens:
                msg_start_usage["cache_creation_input_tokens"] = cache_creation_input_tokens

            # Emit message_start with real token counts from the buffered usage.
            yield StreamEvent(
                event_type="message_start",
                data={
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": msg_start_usage,
                    },
                },
            )

            # Process buffered chunks for content events.
            output_tokens_counted = 0
            current_block_index = -1
            active_block_type: str | None = None  # "text" or "tool_use"
            tool_block_map: dict[int, int] = {}  # litellm tc.index → SSE block index
            stop_reason = "end_turn"

            for chunk in buffered_chunks:
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Check finish_reason to set stop_reason
                if choice.finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif choice.finish_reason == "stop":
                    stop_reason = "end_turn"
                elif choice.finish_reason == "length":
                    stop_reason = "max_tokens"

                # Handle tool_calls in the delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_block_map:
                            # Close previous block if open
                            if active_block_type is not None:
                                yield StreamEvent(
                                    event_type="content_block_stop",
                                    data={
                                        "type": "content_block_stop",
                                        "index": current_block_index,
                                    },
                                )
                            # Open a new tool_use block
                            current_block_index += 1
                            tool_block_map[idx] = current_block_index
                            active_block_type = "tool_use"
                            tool_id = tc.id or f"toolu_{uuid.uuid4().hex[:24]}"
                            tool_name = tc.function.name if tc.function and tc.function.name else ""
                            yield StreamEvent(
                                event_type="content_block_start",
                                data={
                                    "type": "content_block_start",
                                    "index": current_block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_id,
                                        "name": tool_name,
                                        "input": {},
                                    },
                                },
                            )

                        # Emit argument deltas
                        if tc.function and tc.function.arguments:
                            block_idx = tool_block_map[idx]
                            yield StreamEvent(
                                event_type="content_block_delta",
                                data={
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": tc.function.arguments,
                                    },
                                },
                            )
                            output_tokens_counted += 1

                # Handle text content in the delta
                elif hasattr(delta, "content") and delta.content:
                    if active_block_type != "text":
                        # Close previous block if open
                        if active_block_type is not None:
                            yield StreamEvent(
                                event_type="content_block_stop",
                                data={
                                    "type": "content_block_stop",
                                    "index": current_block_index,
                                },
                            )
                        # Open a new text block
                        current_block_index += 1
                        active_block_type = "text"
                        yield StreamEvent(
                            event_type="content_block_start",
                            data={
                                "type": "content_block_start",
                                "index": current_block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )

                    yield StreamEvent(
                        event_type="content_block_delta",
                        data={
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": delta.content},
                        },
                    )
                    output_tokens_counted += 1

            # Close the last open block
            if active_block_type is not None:
                yield StreamEvent(
                    event_type="content_block_stop",
                    data={"type": "content_block_stop", "index": current_block_index},
                )

            # Prefer real completion_tokens from LiteLLM usage; fall back to
            # the per-delta counter when include_usage wasn't honoured.
            output_tokens = output_tokens_from_usage or output_tokens_counted

            # Emit message_delta with correct stop reason
            yield StreamEvent(
                event_type="message_delta",
                data={
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )

            # Emit message_stop
            yield StreamEvent(
                event_type="message_stop",
                data={"type": "message_stop"},
            )

        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            yield StreamEvent(
                event_type="error",
                data={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )

    async def close(self) -> None:  # noqa: B027
        """Clean up (no-op for LiteLLM)."""
        pass

    async def send_openai_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send OpenAI-format message via LiteLLM.

        Unlike send_message(), this takes OpenAI-format input and returns
        OpenAI-format output (no Anthropic conversion).

        Args:
            body: OpenAI chat completion request body
            headers: Request headers (ignored, auth from env vars)

        Returns:
            BackendResponse with OpenAI-format body
        """
        original_model = body.get("model", "gpt-4")
        litellm_model = self.map_model_id(original_model)

        try:
            # Build kwargs - messages already in OpenAI format
            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": body.get("messages", []),
            }

            # Pass through OpenAI parameters
            for param in [
                "max_tokens",
                "temperature",
                "top_p",
                "stop",
                "tools",
                "tool_choice",
                "response_format",
                "seed",
                "n",
            ]:
                if param in body:
                    kwargs[param] = body[param]

            # Provider-specific region config
            if self.region:
                if self.provider == "bedrock":
                    kwargs["aws_region_name"] = self.region
                elif self.provider in ("vertex_ai", "vertex_ai_beta"):
                    kwargs["vertex_location"] = self.region

            # Forward API key from request headers if present.
            # Skip for Bedrock/Vertex: they use env-based auth (AWS SigV4 / Google ADC).
            # Forwarding x-api-key (e.g. sk-ant-dummy) would override their credentials.
            _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
            if self.provider not in _env_auth_providers:
                auth_header = headers.get("authorization", headers.get("Authorization", ""))
                if auth_header.startswith("Bearer "):
                    kwargs["api_key"] = auth_header[7:]
                elif headers.get("x-api-key"):
                    kwargs["api_key"] = headers["x-api-key"]

            logger.debug(f"LiteLLM OpenAI request: model={litellm_model}")

            # Make the call
            response = await acompletion(**kwargs)

            # Build the usage block. LiteLLM normalizes prompt-cache stats from
            # multiple providers (Anthropic, Bedrock-Claude, OpenAI prompt-caching,
            # DeepSeek) onto its Usage object — top-level
            # cache_read_input_tokens / cache_creation_input_tokens for the
            # Anthropic-style dialect, and prompt_tokens_details.cached_tokens /
            # cache_creation_tokens for the OpenAI nested dialect. Surface both
            # so PrefixCacheTracker.update_from_response on the backend-routed
            # path observes a stable shape instead of branching on key presence.
            usage_block: dict[str, Any] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Defensive getattr: LiteLLM only attaches these top-level attrs
            # when the underlying provider returned cache stats. Zero is the
            # cold-start / no-cache value.
            cache_read = int(getattr(response.usage, "cache_read_input_tokens", 0) or 0)
            cache_write = int(getattr(response.usage, "cache_creation_input_tokens", 0) or 0)

            # OpenAI nested dialect — fall back here if the top-level dialect is
            # absent (pure OpenAI prompt caching).
            ptd_obj = getattr(response.usage, "prompt_tokens_details", None)
            ptd_cached = 0
            ptd_cache_creation = 0
            if ptd_obj is not None:
                ptd_cached = int(getattr(ptd_obj, "cached_tokens", 0) or 0)
                ptd_cache_creation = int(getattr(ptd_obj, "cache_creation_tokens", 0) or 0)

            final_cache_read = cache_read or ptd_cached
            final_cache_write = cache_write or ptd_cache_creation

            if final_cache_read or final_cache_write:
                usage_block["cache_read_input_tokens"] = final_cache_read
                usage_block["cache_creation_input_tokens"] = final_cache_write
                # Mirror into the OpenAI nested shape so callers that only know
                # the OpenAI dialect can read it without branching.
                usage_block["prompt_tokens_details"] = {"cached_tokens": final_cache_read}
                logger.debug(
                    f"LiteLLM OpenAI cache stats: cache_read={final_cache_read} "
                    f"cache_write={final_cache_write} model={litellm_model}"
                )

            # Convert ModelResponse to dict (OpenAI format)
            response_dict = {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": original_model,
                "choices": [
                    {
                        "index": c.index,
                        "message": {
                            "role": c.message.role,
                            "content": c.message.content,
                            **(
                                {
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments,
                                            },
                                        }
                                        for tc in c.message.tool_calls
                                    ]
                                }
                                if c.message.tool_calls
                                else {}
                            ),
                        },
                        "finish_reason": c.finish_reason,
                    }
                    for c in response.choices
                ],
                "usage": usage_block,
            }

            return BackendResponse(
                body=response_dict,
                status_code=200,
                headers={"content-type": "application/json"},
            )

        except Exception as e:
            logger.error(f"LiteLLM OpenAI error: {e}")

            # Map to OpenAI error format
            error_type = "api_error"
            status_code = 500

            error_str = str(e).lower()
            if "authentication" in error_str or "credentials" in error_str:
                error_type = "invalid_api_key"
                status_code = 401
            elif "rate" in error_str or "limit" in error_str:
                error_type = "rate_limit_exceeded"
                status_code = 429
            elif "not found" in error_str:
                error_type = "model_not_found"
                status_code = 404

            return BackendResponse(
                body={
                    "error": {
                        "message": str(e),
                        "type": error_type,
                        "code": error_type,
                    }
                },
                status_code=status_code,
                error=str(e),
            )

    async def stream_openai_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """Stream OpenAI-format chat completion via LiteLLM.

        Yields SSE-formatted strings ready to send to the client.
        """
        original_model = body.get("model", "gpt-4")
        litellm_model = self.map_model_id(original_model)

        try:
            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": body.get("messages", []),
                "stream": True,
            }

            for param in [
                "max_tokens",
                "temperature",
                "top_p",
                "stop",
                "tools",
                "tool_choice",
                "response_format",
                "seed",
                "n",
            ]:
                if param in body:
                    kwargs[param] = body[param]

            if "stream_options" in body:
                kwargs["stream_options"] = body["stream_options"]

            # Provider-specific region config
            if self.region:
                if self.provider == "bedrock":
                    kwargs["aws_region_name"] = self.region
                elif self.provider in ("vertex_ai", "vertex_ai_beta"):
                    kwargs["vertex_location"] = self.region

            # Forward API key from request headers if present.
            # Skip for Bedrock/Vertex: they use env-based auth (AWS SigV4 / Google ADC).
            # Forwarding x-api-key (e.g. sk-ant-dummy) would override their credentials.
            _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
            if self.provider not in _env_auth_providers:
                auth_header = headers.get("authorization", headers.get("Authorization", ""))
                if auth_header.startswith("Bearer "):
                    kwargs["api_key"] = auth_header[7:]
                elif headers.get("x-api-key"):
                    kwargs["api_key"] = headers["x-api-key"]

            response = await acompletion(**kwargs)

            async for chunk in response:
                chunk_dict = chunk.model_dump(exclude_none=True, exclude_unset=True)
                yield f"data: {json.dumps(chunk_dict)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"LiteLLM OpenAI streaming error: {e}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "backend_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
