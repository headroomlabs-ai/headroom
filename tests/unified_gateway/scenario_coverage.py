"""Concrete executable evidence for the retained T001--T100 scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CoverageCell:
    status: Literal["local_test", "unavailable_test", "external_not_run"]
    test_nodes: tuple[str, ...]
    note: str = ""


def _local(*nodes: str) -> CoverageCell:
    return CoverageCell("local_test", nodes)


def _external(note: str) -> CoverageCell:
    return CoverageCell("external_not_run", (), note)


def _unavailable(*nodes: str) -> CoverageCell:
    return CoverageCell("unavailable_test", nodes)


_COVERAGE: dict[str, CoverageCell] = {
    "T001": _local(
        "tests/test_cli_proxy_improvements.py::TestLearnNoLearnConflict::test_no_learn_alone_disables_learning"
    ),
    "T002": _local(
        "tests/unified_gateway/test_cli_profile.py::test_gateway_rejects_transforming_flags"
    ),
    "T003": _local(
        "tests/unified_gateway/test_config.py::test_redacted_dict_contains_references_not_resolved_secret_values"
    ),
    "T004": _local(
        "tests/unified_gateway/test_config.py::test_check_config_new_policies_has_no_readers"
    ),
    "T005": _unavailable("tests/unified_gateway/test_cli_profile.py::test_gateway_requires_config"),
    "T006": _local(
        "tests/unified_gateway/process/test_protocol_completeness_runtime.py::test_strict_native_signed_and_tool_entities_are_byte_exact_over_tls"
    ),
    "T007": _local(
        "tests/unified_gateway/test_pure_profile.py::test_gateway_profile_disables_every_hidden_transform"
    ),
    "T008": _local(
        "tests/unified_gateway/process/test_protocol_completeness_runtime.py::test_strict_native_signed_and_tool_entities_are_byte_exact_over_tls"
    ),
    "T009": _local(
        "tests/unified_gateway/test_native_bytes.py::test_routed_native_changes_only_model_field"
    ),
    "T010": _local(
        "tests/unified_gateway/test_pure_profile.py::test_proxy_config_applies_gateway_profile_for_non_cli_callers"
    ),
    "T011": _local(
        "tests/unified_gateway/process/test_no_egress.py::test_unauthenticated_request_is_rejected_without_egress"
    ),
    "T012": _local(
        "tests/unified_gateway/test_auth.py::test_real_gateway_app_requires_auth_even_on_loopback"
    ),
    "T013": _local(
        "tests/unified_gateway/test_auth.py::test_conflicting_protocol_credentials_are_rejected"
    ),
    "T014": _local(
        "tests/unified_gateway/process/test_sdk_openai.py::test_openai_sdk_lists_only_principal_visible_models"
    ),
    "T015": _local(
        "tests/unified_gateway/test_control.py::test_admin_controls_are_scoped_and_reload_uses_startup_path"
    ),
    "T016": _local(
        "tests/unified_gateway/test_egress.py::test_client_auth_headers_are_removed_before_provider_credential_is_added"
    ),
    "T017": _local(
        "tests/unified_gateway/test_egress.py::test_origin_path_and_userinfo_confusion_are_denied"
    ),
    "T018": _local(
        "tests/unified_gateway/test_security_boundary.py::test_config_rejects_incompatible_provider_audience"
    ),
    "T019": _local(
        "tests/unified_gateway/test_security_boundary.py::test_egress_without_explicit_addresses_resolves_and_denies_metadata",
        "tests/unified_gateway/test_security_boundary.py::test_egress_rejects_ambiguous_final_path",
    ),
    "T020": _local(
        "tests/unified_gateway/test_security_boundary.py::test_http_transport_pins_ip_and_retains_tls_hostname"
    ),
    "T021": _local(
        "tests/unified_gateway/test_refresh.py::test_concurrent_acquisition_is_single_flight"
    ),
    "T022": _local(
        "tests/unified_gateway/test_security_boundary.py::test_stale_invalidation_cannot_evict_newer_generation"
    ),
    "T023": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow"
    ),
    "T024": _local(
        "tests/unified_gateway/process/test_websocket_runtime.py::test_affinity_does_not_migrate_when_owner_cools"
    ),
    "T025": _external(
        "real cloud credential expiry and refresh requires an approved live identity"
    ),
    "T026": _local("tests/unified_gateway/test_model_catalog.py::test_alias_collision_rejected"),
    "T027": _local(
        "tests/unified_gateway/process/test_runtime_catalog.py::test_denied_entitlement_hides_model_and_blocks_generation"
    ),
    "T028": _local(
        "tests/unified_gateway/process/test_runtime_catalog.py::test_failed_catalog_refresh_is_explicitly_stale_then_unavailable"
    ),
    "T029": _local(
        "tests/unified_gateway/process/test_runtime_catalog.py::test_capability_rejection_has_zero_identity_or_generation_calls"
    ),
    "T030": _local(
        "tests/unified_gateway/process/test_runtime_catalog.py::test_inflight_catalog_refresh_preserves_metadata_and_tariff"
    ),
    "T031": _unavailable(
        "tests/unified_gateway/test_translation.py::test_public_parallel_tool_translation_is_unavailable"
    ),
    "T032": _local(
        "tests/unified_gateway/test_translation.py::test_inline_image_and_text_order_survives_openai_to_anthropic"
    ),
    "T033": _unavailable(
        "tests/unified_gateway/test_protocol_completeness.py::test_openai_unrepresentable_request_rejects_without_reordering"
    ),
    "T034": _local(
        "tests/unified_gateway/test_translation.py::test_unrepresentable_fields_are_rejected"
    ),
    "T035": _local(
        "tests/unified_gateway/test_translation.py::test_response_translation_matches_literal_finish_and_usage_oracle"
    ),
    "T036": _local(
        "tests/unified_gateway/process/test_streaming_runtime.py::test_first_event_precedes_upstream_completion_barrier"
    ),
    "T037": _local(
        "tests/unified_gateway/process/test_streaming_runtime.py::test_fragmented_unicode_tool_events_cross_real_sockets"
    ),
    "T038": _local(
        "tests/unified_gateway/process/test_streaming_runtime.py::test_eof_without_terminal_is_failed"
    ),
    "T039": _local(
        "tests/unified_gateway/process/test_streaming_runtime.py::test_disconnect_cancels_upstream_and_finalizes_once"
    ),
    "T040": _local(
        "tests/unified_gateway/process/test_streaming_runtime.py::test_heartbeat_and_partial_line_deadlines_fire_without_new_bytes"
    ),
    "T041": _local(
        "tests/unified_gateway/process/test_stateful_runtime.py::test_other_principal_cannot_read_continue_cancel_delete"
    ),
    "T042": _local(
        "tests/unified_gateway/process/test_stateful_runtime.py::test_affinity_does_not_migrate_when_owner_unavailable"
    ),
    "T043": _local(
        "tests/unified_gateway/process/test_websocket_runtime.py::test_second_turn_exhausted_budget_is_not_forwarded"
    ),
    "T044": _local(
        "tests/unified_gateway/process/test_websocket_runtime.py::test_accepted_frame_disconnect_never_replays"
    ),
    "T045": _local(
        "tests/unified_gateway/process/test_stateful_runtime.py::test_all_route_and_upgrade_variants_enforce_runtime"
    ),
    "T046": _local(
        "tests/unified_gateway/process/test_retry_runtime.py::test_proven_connect_failure_retries_once_under_same_deadline"
    ),
    "T047": _local(
        "tests/unified_gateway/process/test_retry_runtime.py::test_body_sent_disconnect_is_ambiguous_and_not_retried"
    ),
    "T048": _local(
        "tests/unified_gateway/process/test_retry_runtime.py::test_output_then_quota_error_never_switches_account"
    ),
    "T049": _local(
        "tests/unified_gateway/process/test_retry_runtime.py::test_cooldown_uses_quota_scope_and_equivalent_billing_only"
    ),
    "T050": _local(
        "tests/unified_gateway/process/test_retry_runtime.py::test_overload_storm_respects_attempt_queue_and_deadline_bounds"
    ),
    "T051": _local(
        "tests/unified_gateway/process/test_admission_runtime.py::test_two_routes_race_one_remaining_budget_reservation"
    ),
    "T052": _local(
        "tests/unified_gateway/process/test_admission_runtime.py::test_retry_accounting_keeps_known_and_unknown_attempts"
    ),
    "T053": _local(
        "tests/unified_gateway/process/test_admission_runtime.py::test_terminal_path_leak_matrix"
    ),
    "T054": _local(
        "tests/unified_gateway/process/test_admission_runtime.py::test_noisy_tenant_cannot_consume_reserved_other_tenant_slots"
    ),
    "T055": _local(
        "tests/unified_gateway/test_usage.py::test_unknown_cost_and_allowance_units_are_not_zero_usd"
    ),
    "T056": _local(
        "tests/unified_gateway/test_security_boundary.py::test_real_dispatch_never_reflects_provider_error_or_secret_header"
    ),
    "T057": _local(
        "tests/unified_gateway/test_security_boundary.py::test_gateway_startup_does_not_configure_inherited_external_telemetry"
    ),
    "T058": _local(
        "tests/unified_gateway/test_execution.py::test_observability_rejects_arbitrary_labels_and_separates_event_kinds"
    ),
    "T059": _local(
        "tests/unified_gateway/test_config.py::test_check_config_new_policies_has_no_readers"
    ),
    "T060": _local(
        "tests/unified_gateway/test_security_boundary.py::test_dispatch_sanitizes_late_body_and_source_exceptions"
    ),
    "T061": _local(
        "tests/unified_gateway/test_streaming.py::test_malformed_and_oversized_frames_fail",
        "tests/unified_gateway/test_security_boundary.py::test_egress_rejects_ambiguous_final_path",
    ),
    "T062": _local(
        "tests/unified_gateway/test_egress.py::test_public_credentials_cannot_reach_private_or_metadata_addresses"
    ),
    "T063": _local(
        "tests/unified_gateway/test_browser_guards.py::test_hostile_browser_or_forwarded_headers_are_rejected"
    ),
    "T064": _local(
        "tests/unified_gateway/test_browser_guards.py::test_hostile_browser_or_forwarded_headers_are_rejected"
    ),
    "T065": _unavailable(
        "tests/unified_gateway/test_capability_policy.py::test_protocol_feature_rejected_before_identity_and_dispatch"
    ),
    "T066": _local(
        "tests/unified_gateway/test_reload.py::test_invalid_reload_keeps_previous_snapshot",
        "tests/unified_gateway/test_reload.py::test_valid_reload_publishes_one_complete_generation",
    ),
    "T067": _local(
        "tests/unified_gateway/process/test_reload_runtime.py::test_revoke_denies_waiters_and_cancels_http_ws_and_acquire"
    ),
    "T068": _local(
        "tests/unified_gateway/process/test_shutdown_runtime.py::test_only_owned_children_and_sockets_are_closed"
    ),
    "T069": _local(
        "tests/unified_gateway/process/test_shutdown_runtime.py::test_wrong_listener_200_cannot_satisfy_startup"
    ),
    "T070": _local(
        "tests/unified_gateway/process/test_control_runtime.py::test_liveness_readiness_and_status_do_not_acquire_expired_source"
    ),
    "T071": _external("real desktop credential stores require approved platform qualification"),
    "T072": _local(
        "tests/unified_gateway/test_native_dispatch.py::test_cloud_native_routes_use_gateway_identity_and_exact_target"
    ),
    "T073": _external(
        "managed enterprise source precedence requires approved live platform access"
    ),
    "T074": _unavailable(
        "tests/unified_gateway/test_security_boundary.py::test_config_rejects_every_unadmitted_provider_source_pair"
    ),
    "T075": _external("locked OS keychain behavior requires approved live platform access"),
    "T076": _local(
        "tests/unified_gateway/process/test_sdk_openai.py::test_openai_sdk_lists_only_principal_visible_models",
        "tests/unified_gateway/process/test_sdk_anthropic.py::test_anthropic_sdk_receives_local_unknown_model_error",
        "tests/unified_gateway/process/test_sdk_gemini.py::test_gemini_native_shape_is_rejected_locally_for_unknown_model",
    ),
    "T077": _local(
        "tests/test_cli_proxy_improvements.py::TestLearnNoLearnConflict::test_no_learn_alone_disables_learning"
    ),
    "T078": _unavailable("tests/unified_gateway/test_cli_profile.py::test_gateway_requires_config"),
    "T079": _local(
        "tests/test_mcp_dependency_contract.py::test_all_shipping_mcp_extras_exclude_incompatible_sdk_v2"
    ),
    "T080": _unavailable(
        "tests/unified_gateway/test_translation.py::test_public_tool_use_and_result_translation_is_unavailable"
    ),
    "T081": _external("Gate A must be rerun against the final retained release source"),
    "T082": _external("Gate B requires an official retained release artifact"),
    "T083": _local(
        "tests/test_rollout.py::test_unsafe_override_crosses_channel_and_poisons_qualification"
    ),
    "T084": _local(
        "tests/unified_gateway/test_scenario_coverage.py::test_every_scenario_has_an_executable_or_explicit_external_cell"
    ),
    "T085": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow"
    ),
    "T086": _local(
        "tests/unified_gateway/test_provider_admission.py::test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow"
    ),
    "T087": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_unadmitted_native_provider_is_not_advertised"
    ),
    "T088": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_unadmitted_native_provider_is_not_advertised"
    ),
    "T089": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_unadmitted_native_provider_is_not_advertised"
    ),
    "T090": _local(
        "tests/unified_gateway/test_provider_admission.py::test_identifying_header_impersonation_is_closed"
    ),
    "T091": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow"
    ),
    "T092": _local(
        "tests/unified_gateway/test_oauth.py::test_browser_transaction_uses_pkce_s256_and_exact_callback"
    ),
    "T093": _local(
        "tests/unified_gateway/test_oauth.py::test_callback_rejects_lookalikes_and_mismatched_binding"
    ),
    "T094": _local("tests/unified_gateway/test_oauth.py::test_callback_state_is_consumed_once"),
    "T095": _local(
        "tests/unified_gateway/test_device_flow.py::test_device_flow_honors_pending_and_slow_down",
        "tests/unified_gateway/test_device_flow.py::test_device_flow_denial_is_terminal",
    ),
    "T096": _local(
        "tests/unified_gateway/test_provider_admission.py::test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow"
    ),
    "T097": _local(
        "tests/unified_gateway/test_oauth.py::test_callback_rejects_lookalikes_and_mismatched_binding"
    ),
    "T098": _local(
        "tests/unified_gateway/test_device_flow.py::test_device_flow_cancellation_stops_before_polling"
    ),
    "T099": _unavailable(
        "tests/unified_gateway/test_provider_admission.py::test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow"
    ),
    "T100": _local(
        "tests/unified_gateway/test_provider_admission.py::test_identifying_header_impersonation_is_closed"
    ),
}


def load_scenario_coverage() -> dict[str, CoverageCell]:
    return dict(_COVERAGE)
