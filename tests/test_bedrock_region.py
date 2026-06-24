"""Tests for Bedrock region support and fallback model mapping.

Ensures that EU, AP, and US regions all produce valid Bedrock model IDs,
and that the proxy degrades gracefully when boto3 is unavailable or the
AWS API call fails.
"""

from unittest.mock import MagicMock, patch

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import (  # noqa: E402  (must follow importorskip)
    LiteLLMBackend,
    _bedrock_profiles_cache,
    _bedrock_region_prefix,
    _build_bedrock_fallback_map,
    _fetch_bedrock_inference_profiles,
    _normalize_bedrock_profile_id,
)

# =============================================================================
# Region Prefix Mapping
# =============================================================================


class TestBedrockRegionPrefix:
    """Test AWS region -> inference profile prefix mapping."""

    def test_us_regions(self):
        assert _bedrock_region_prefix("us-east-1") == "us"
        assert _bedrock_region_prefix("us-west-2") == "us"

    def test_eu_regions(self):
        assert _bedrock_region_prefix("eu-central-1") == "eu"
        assert _bedrock_region_prefix("eu-west-1") == "eu"
        assert _bedrock_region_prefix("eu-west-3") == "eu"
        assert _bedrock_region_prefix("eu-north-1") == "eu"

    def test_ap_regions(self):
        assert _bedrock_region_prefix("ap-southeast-1") == "apac"
        assert _bedrock_region_prefix("ap-northeast-1") == "apac"

    def test_unknown_region_defaults_to_us(self):
        assert _bedrock_region_prefix("me-south-1") == "us"
        assert _bedrock_region_prefix("sa-east-1") == "us"


# =============================================================================
# Static Fallback Model Map
# =============================================================================


class TestBuildBedrockFallbackMap:
    """Test static fallback model map construction."""

    def test_us_region_uses_us_prefix(self):
        model_map = _build_bedrock_fallback_map("us-east-1")
        assert "claude-sonnet-4-20250514" in model_map
        assert model_map["claude-sonnet-4-20250514"] == (
            "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
        )

    def test_eu_region_uses_eu_prefix(self):
        model_map = _build_bedrock_fallback_map("eu-central-1")
        assert "claude-sonnet-4-20250514" in model_map
        assert model_map["claude-sonnet-4-20250514"] == (
            "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"
        )

    def test_ap_region_uses_apac_prefix(self):
        model_map = _build_bedrock_fallback_map("ap-southeast-1")
        assert "claude-sonnet-4-20250514" in model_map
        assert model_map["claude-sonnet-4-20250514"] == (
            "bedrock/apac.anthropic.claude-sonnet-4-20250514-v1:0"
        )

    def test_all_models_present(self):
        model_map = _build_bedrock_fallback_map("us-east-1")
        expected_models = [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-haiku-4-5-20251001",
        ]
        for model in expected_models:
            assert model in model_map, f"Missing model: {model}"

    def test_all_values_are_valid_bedrock_format(self):
        """Every value must start with 'bedrock/' and contain 'anthropic.'."""
        for region in ("us-east-1", "eu-west-1", "ap-northeast-1"):
            model_map = _build_bedrock_fallback_map(region)
            for name, litellm_id in model_map.items():
                assert litellm_id.startswith("bedrock/"), (
                    f"{name}: expected bedrock/ prefix, got {litellm_id}"
                )
                assert "anthropic." in litellm_id, (
                    f"{name}: expected anthropic. in id, got {litellm_id}"
                )


# =============================================================================
# Fetch with Graceful Fallback
# =============================================================================


class TestFetchBedrockInferenceProfiles:
    """Test dynamic fetch with fallback on failure."""

    def setup_method(self):
        """Clear the cache before each test."""
        _bedrock_profiles_cache.clear()

    def test_fallback_when_boto3_import_fails(self):
        """Should return static map when boto3 is not installed."""
        with patch.dict("sys.modules", {"boto3": None}):
            # Force reimport failure

            # Temporarily break boto3 import inside the function
            original_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "boto3":
                    raise ImportError("No module named 'boto3'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                _bedrock_profiles_cache.clear()
                result = _fetch_bedrock_inference_profiles("eu-central-1")

            assert len(result) > 0
            # Should use EU prefix
            assert result["claude-sonnet-4-20250514"] == (
                "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"
            )

    def test_fallback_when_api_call_fails(self):
        """Should return static map when list_inference_profiles raises."""
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.list_inference_profiles.side_effect = Exception(
            "AccessDeniedException: not authorized"
        )
        mock_boto3.client.return_value = mock_client

        with patch("headroom.backends.litellm.boto3", mock_boto3, create=True):
            # Patch the import inside the function
            _fetch_bedrock_inference_profiles.__code__  # noqa: B018
            _bedrock_profiles_cache.clear()

            # We need to actually test the function, so let's just use the
            # mock_boto3 and make sure the function catches the exception
            import builtins

            real_import = builtins.__import__

            def patched_import(name, *args, **kwargs):
                if name == "boto3":
                    return mock_boto3
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=patched_import):
                _bedrock_profiles_cache.clear()
                result = _fetch_bedrock_inference_profiles("eu-west-1")

            assert len(result) > 0
            # Should use EU prefix
            for litellm_id in result.values():
                assert "eu.anthropic." in litellm_id

    def test_successful_fetch_uses_api_results(self):
        """When API works, should use dynamic results (not fallback)."""
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {"inferenceProfileId": "eu.anthropic.claude-sonnet-4-20250514-v1:0"},
                {"inferenceProfileId": "eu.anthropic.claude-3-5-sonnet-20241022-v2:0"},
                {"inferenceProfileId": "eu.meta.llama-3-70b-v1:0"},  # non-Anthropic, should skip
            ]
        }
        mock_boto3.client.return_value = mock_client

        import builtins

        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "boto3":
                return mock_boto3
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            _bedrock_profiles_cache.clear()
            result = _fetch_bedrock_inference_profiles("eu-central-1")

        assert len(result) == 2
        assert result["claude-sonnet-4-20250514"] == (
            "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        assert result["claude-3-5-sonnet-20241022"] == (
            "bedrock/eu.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

    def test_caching_prevents_repeated_api_calls(self):
        """Second call for same region should return cached result."""
        _bedrock_profiles_cache.clear()
        _bedrock_profiles_cache["us-east-1"] = {"test": "bedrock/test-model"}

        result = _fetch_bedrock_inference_profiles("us-east-1")
        assert result == {"test": "bedrock/test-model"}


# =============================================================================
# LiteLLMBackend.map_model_id with EU Regions
# =============================================================================


class TestBedrockModelMapping:
    """Test model ID mapping for different regions."""

    def setup_method(self):
        _bedrock_profiles_cache.clear()

    def test_eu_region_maps_correctly(self):
        """EU region should produce eu.anthropic.* model IDs."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={
                "claude-sonnet-4-20250514": "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
            },
        ):
            backend = LiteLLMBackend(provider="bedrock", region="eu-central-1")
            result = backend.map_model_id("claude-sonnet-4-20250514")
            assert result == "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_us_region_maps_correctly(self):
        """US region should produce us.anthropic.* model IDs."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={
                "claude-sonnet-4-20250514": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
            },
        ):
            backend = LiteLLMBackend(provider="bedrock", region="us-west-2")
            result = backend.map_model_id("claude-sonnet-4-20250514")
            assert result == "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_fallback_for_unknown_model_in_eu(self):
        """Unknown models in EU should get eu.anthropic.* fallback, not bare 'bedrock/claude-...'."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={},
        ):
            backend = LiteLLMBackend(provider="bedrock", region="eu-west-1")
            result = backend.map_model_id("claude-sonnet-4-20250514")
            assert result == "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_fallback_for_unknown_model_in_ap(self):
        """Unknown models in AP should get apac.anthropic.* fallback."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={},
        ):
            backend = LiteLLMBackend(provider="bedrock", region="ap-southeast-1")
            result = backend.map_model_id("claude-3-5-haiku-20241022")
            assert result == "bedrock/apac.anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_bedrock_format_passthrough(self):
        """Already-formatted Bedrock IDs should pass through unchanged."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={},
        ):
            backend = LiteLLMBackend(provider="bedrock", region="eu-central-1")
            model = "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"
            result = backend.map_model_id(model)
            assert result == model

    def test_anthropic_dot_format_normalized(self):
        """Raw Bedrock IDs like 'anthropic.claude-...-v1:0' should normalize and map."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={
                "claude-sonnet-4-20250514": "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
            },
        ):
            backend = LiteLLMBackend(provider="bedrock", region="eu-central-1")
            result = backend.map_model_id("anthropic.claude-sonnet-4-20250514-v1:0")
            assert result == "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_region_prefixed_format_normalized(self):
        """'eu.anthropic.claude-...-v1:0' should normalize and map."""
        with patch(
            "headroom.backends.litellm._fetch_bedrock_inference_profiles",
            return_value={
                "claude-sonnet-4-20250514": "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
            },
        ):
            backend = LiteLLMBackend(provider="bedrock", region="eu-central-1")
            result = backend.map_model_id("eu.anthropic.claude-sonnet-4-20250514-v1:0")
            assert result == "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0"


# =============================================================================
# Normalize Bedrock Profile ID (edge cases)
# =============================================================================


class TestNormalizeBedrockProfileId:
    """Test normalization of various Bedrock profile ID formats."""

    def test_eu_prefixed(self):
        assert _normalize_bedrock_profile_id("eu.anthropic.claude-sonnet-4-20250514-v1:0") == (
            "claude-sonnet-4-20250514"
        )

    def test_apac_prefixed(self):
        assert _normalize_bedrock_profile_id("apac.anthropic.claude-3-5-sonnet-20241022-v2:0") == (
            "claude-3-5-sonnet-20241022"
        )

    def test_us_prefixed(self):
        assert _normalize_bedrock_profile_id("us.anthropic.claude-opus-4-20250514-v1:0") == (
            "claude-opus-4-20250514"
        )

    def test_no_region_prefix(self):
        assert _normalize_bedrock_profile_id("anthropic.claude-3-haiku-20240307-v1:0") == (
            "claude-3-haiku-20240307"
        )

    def test_with_bedrock_slash_prefix(self):
        assert (
            _normalize_bedrock_profile_id("bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0")
            == "claude-sonnet-4-20250514"
        )

    def test_non_claude_returns_none(self):
        assert _normalize_bedrock_profile_id("eu.meta.llama-3-70b-v1:0") is None

    def test_already_normalized(self):
        assert _normalize_bedrock_profile_id("claude-sonnet-4-20250514") == (
            "claude-sonnet-4-20250514"
        )
