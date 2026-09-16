//! Deterministic runtime-rollout policy and provenance.
//!
//! Rollout channels control behavior in an already-built artifact. They do not
//! select a package, release candidate, or distribution version. Composition
//! roots resolve one immutable snapshot and inject its concrete decisions.

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::str::FromStr;

pub const ROLLOUT_SCHEMA_VERSION: u32 = 1;
pub const ROLLOUT_POLICY_VERSION: &str = "1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutChannel {
    #[default]
    Stable,
    Beta,
    Canary,
    Dev,
}

impl RolloutChannel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
            Self::Canary => "canary",
            Self::Dev => "dev",
        }
    }

    pub fn allows(self, required: Self) -> bool {
        self >= required
    }
}

impl FromStr for RolloutChannel {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "" | "stable" | "prod" | "production" => Ok(Self::Stable),
            "beta" | "preview" => Ok(Self::Beta),
            "canary" | "nightly" => Ok(Self::Canary),
            "dev" | "development" => Ok(Self::Dev),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    NativeBedrock,
    OpenAiResponsesStreaming,
    CanaryProbe,
}

const ALL_FEATURES: [Feature; 3] = [
    Feature::CanaryProbe,
    Feature::NativeBedrock,
    Feature::OpenAiResponsesStreaming,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FeatureSpec {
    pub name: &'static str,
    pub available_in: RolloutChannel,
    pub default_enabled_in: Option<RolloutChannel>,
}

impl Feature {
    pub fn spec(self) -> FeatureSpec {
        match self {
            Self::NativeBedrock => FeatureSpec {
                name: "native_bedrock",
                available_in: RolloutChannel::Stable,
                default_enabled_in: Some(RolloutChannel::Stable),
            },
            Self::OpenAiResponsesStreaming => FeatureSpec {
                name: "openai_responses_streaming",
                available_in: RolloutChannel::Stable,
                default_enabled_in: Some(RolloutChannel::Stable),
            },
            Self::CanaryProbe => FeatureSpec {
                name: "canary_probe",
                available_in: RolloutChannel::Canary,
                default_enabled_in: None,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureDecisionReason {
    Default,
    Explicit,
    LegacyAlias,
    Disabled,
    BlockedByChannel,
    UnsafeOverride,
    NotRequested,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RolloutConfig {
    pub channel: RolloutChannel,
    pub requested: BTreeSet<String>,
    pub disabled: BTreeSet<String>,
    pub unsafe_allow_unstable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FeatureDecision {
    pub name: &'static str,
    pub available_in: RolloutChannel,
    pub default_enabled_in: Option<RolloutChannel>,
    pub requested: bool,
    pub disabled: bool,
    pub enabled: bool,
    #[serde(rename = "decision")]
    pub reason: FeatureDecisionReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutSnapshot {
    pub schema_version: u32,
    pub policy_version: &'static str,
    pub registry_digest: String,
    pub config: RolloutConfig,
    pub decisions: Vec<FeatureDecision>,
}

impl Default for RolloutSnapshot {
    fn default() -> Self {
        Self::from_parts("stable", "", "", false)
    }
}

impl RolloutSnapshot {
    pub fn from_parts(
        channel: &str,
        requested: &str,
        disabled: &str,
        unsafe_allow_unstable: bool,
    ) -> Self {
        Self::from_parts_with_explicit(channel, requested, disabled, unsafe_allow_unstable, &[])
    }

    pub fn from_parts_with_explicit(
        channel: &str,
        requested: &str,
        disabled: &str,
        unsafe_allow_unstable: bool,
        explicit: &[Feature],
    ) -> Self {
        let parsed_channel = RolloutChannel::from_str(channel).unwrap_or_else(|_| {
            tracing::warn!(channel, "unknown rollout channel; falling back to stable");
            RolloutChannel::Stable
        });
        let valid_names: BTreeSet<_> = ALL_FEATURES
            .iter()
            .map(|feature| feature.spec().name.to_owned())
            .collect();
        let mut requested_names = validated_names(requested, "requested", &valid_names);
        requested_names.extend(
            explicit
                .iter()
                .map(|feature| feature.spec().name.to_owned()),
        );
        let disabled_names = validated_names(disabled, "disabled", &valid_names);
        let config = RolloutConfig {
            channel: parsed_channel,
            requested: requested_names,
            disabled: disabled_names,
            unsafe_allow_unstable,
        };
        let decisions = ALL_FEATURES
            .iter()
            .map(|feature| resolve_feature(*feature, &config))
            .collect();
        Self {
            schema_version: ROLLOUT_SCHEMA_VERSION,
            policy_version: ROLLOUT_POLICY_VERSION,
            registry_digest: registry_digest(),
            config,
            decisions,
        }
    }

    pub fn decision(&self, feature: Feature) -> &FeatureDecision {
        let name = feature.spec().name;
        self.decisions
            .iter()
            .find(|decision| decision.name == name)
            .expect("every registered feature has a decision")
    }

    pub fn is_enabled(&self, feature: Feature, _explicit: bool) -> bool {
        self.decision(feature).enabled
    }

    pub fn enabled(&self) -> BTreeSet<String> {
        self.decisions
            .iter()
            .filter(|decision| decision.enabled)
            .map(|decision| decision.name.to_owned())
            .collect()
    }

    pub fn qualification_eligible(&self) -> bool {
        !self.config.unsafe_allow_unstable
    }

    fn canonical_value(&self) -> Value {
        json!({
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "channel": self.config.channel,
            "unsafe_override": self.config.unsafe_allow_unstable,
            "registry_digest": self.registry_digest,
            "features": self.decisions,
        })
    }

    pub fn snapshot_digest(&self) -> String {
        digest_value(&self.canonical_value())
    }

    pub fn to_value(&self) -> Value {
        let mut value = self.canonical_value();
        let object = value
            .as_object_mut()
            .expect("rollout snapshot is an object");
        object.insert("snapshot_digest".into(), json!(self.snapshot_digest()));
        object.insert(
            "qualification_eligible".into(),
            json!(self.qualification_eligible()),
        );
        if !self.qualification_eligible() {
            object.insert(
                "qualification_ineligible_reason".into(),
                json!("unsafe_rollout_override_active"),
            );
        }
        value
    }
}

fn resolve_feature(feature: Feature, config: &RolloutConfig) -> FeatureDecision {
    let spec = feature.spec();
    let requested = config.requested.contains(spec.name);
    let disabled = config.disabled.contains(spec.name);
    let normally_available = config.channel.allows(spec.available_in);
    let (enabled, reason) = if disabled {
        (false, FeatureDecisionReason::Disabled)
    } else if requested && !normally_available && !config.unsafe_allow_unstable {
        (false, FeatureDecisionReason::BlockedByChannel)
    } else if requested && !normally_available {
        (true, FeatureDecisionReason::UnsafeOverride)
    } else if requested {
        (true, FeatureDecisionReason::Explicit)
    } else if spec
        .default_enabled_in
        .is_some_and(|minimum| config.channel.allows(minimum))
    {
        (true, FeatureDecisionReason::Default)
    } else {
        (false, FeatureDecisionReason::NotRequested)
    };
    FeatureDecision {
        name: spec.name,
        available_in: spec.available_in,
        default_enabled_in: spec.default_enabled_in,
        requested,
        disabled,
        enabled,
        reason,
    }
}

fn validated_names(raw: &str, source: &str, valid: &BTreeSet<String>) -> BTreeSet<String> {
    let names: BTreeSet<_> = split_feature_names(raw).into_iter().collect();
    for unknown in names.difference(valid) {
        tracing::warn!(
            feature = unknown,
            source,
            "unknown rollout feature; ignoring (fail-closed)"
        );
    }
    names.intersection(valid).cloned().collect()
}

pub fn split_feature_names(raw: &str) -> Vec<String> {
    raw.replace(';', ",")
        .split(',')
        .filter_map(|part| {
            let normalized = normalize_feature_name(part);
            (!normalized.is_empty()).then_some(normalized)
        })
        .collect()
}

pub fn normalize_feature_name(raw: impl AsRef<str>) -> String {
    raw.as_ref().trim().to_ascii_lowercase().replace('-', "_")
}

pub fn registry_digest() -> String {
    let registry: Vec<_> = ALL_FEATURES.iter().map(|feature| feature.spec()).collect();
    digest_value(&serde_json::to_value(registry).expect("registry is serializable"))
}

pub fn feature_names() -> BTreeSet<&'static str> {
    ALL_FEATURES
        .iter()
        .map(|feature| feature.spec().name)
        .collect()
}

fn digest_value(value: &Value) -> String {
    let canonical = serde_json::to_vec(value).expect("rollout provenance is serializable");
    let digest = Sha256::digest(canonical);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    format!("sha256:{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct PolicyVector {
        channel: String,
        requested: bool,
        disabled: bool,
        #[serde(rename = "unsafe")]
        unsafe_override: bool,
        enabled: bool,
        decision: String,
    }

    #[test]
    fn channel_order_matches_python_policy() {
        assert!(RolloutChannel::Dev.allows(RolloutChannel::Canary));
        assert!(RolloutChannel::Canary.allows(RolloutChannel::Beta));
        assert!(!RolloutChannel::Stable.allows(RolloutChannel::Canary));
    }

    #[test]
    fn stable_blocks_explicit_canary_feature_with_reason() {
        let rollout = RolloutSnapshot::from_parts("stable", "canary_probe", "", false);
        let decision = rollout.decision(Feature::CanaryProbe);
        assert!(!decision.enabled);
        assert_eq!(decision.reason, FeatureDecisionReason::BlockedByChannel);
    }

    #[test]
    fn default_enabled_feature_has_default_reason() {
        let rollout = RolloutSnapshot::default();
        let decision = rollout.decision(Feature::NativeBedrock);
        assert!(decision.enabled);
        assert_eq!(decision.reason, FeatureDecisionReason::Default);
    }

    #[test]
    fn unsafe_override_crosses_boundary_and_is_ineligible() {
        let rollout = RolloutSnapshot::from_parts("stable", "canary_probe", "", true);
        assert_eq!(
            rollout.decision(Feature::CanaryProbe).reason,
            FeatureDecisionReason::UnsafeOverride
        );
        assert!(!rollout.qualification_eligible());
        assert_eq!(
            rollout.to_value()["qualification_ineligible_reason"],
            "unsafe_rollout_override_active"
        );
    }

    #[test]
    fn disable_beats_default_explicit_and_unsafe() {
        for unsafe_override in [false, true] {
            let rollout = RolloutSnapshot::from_parts(
                "stable",
                "native_bedrock",
                "native-bedrock",
                unsafe_override,
            );
            assert_eq!(
                rollout.decision(Feature::NativeBedrock).reason,
                FeatureDecisionReason::Disabled
            );
        }
    }

    #[test]
    fn provenance_digests_are_deterministic_and_policy_sensitive() {
        let first = RolloutSnapshot::from_parts("canary", "canary_probe", "", false);
        let second = RolloutSnapshot::from_parts("canary", "canary_probe", "", false);
        let changed = RolloutSnapshot::from_parts("stable", "canary_probe", "", false);
        assert_eq!(first.registry_digest, second.registry_digest);
        assert_eq!(first.snapshot_digest(), second.snapshot_digest());
        assert_ne!(first.snapshot_digest(), changed.snapshot_digest());
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let rollout = RolloutSnapshot::from_parts("stabel", "unknown", "unknown", false);
        assert_eq!(rollout.config.channel, RolloutChannel::Stable);
        assert!(rollout.config.requested.is_empty());
        assert!(rollout.config.disabled.is_empty());
    }

    #[test]
    fn shared_python_rust_policy_vectors() {
        let vectors: Vec<PolicyVector> = serde_json::from_str(include_str!(
            "../../../tests/fixtures/rollout_policy_vectors.json"
        ))
        .unwrap();
        for vector in vectors {
            let requested = if vector.requested { "canary_probe" } else { "" };
            let disabled = if vector.disabled { "canary_probe" } else { "" };
            let rollout = RolloutSnapshot::from_parts(
                &vector.channel,
                requested,
                disabled,
                vector.unsafe_override,
            );
            let decision = rollout.decision(Feature::CanaryProbe);
            assert_eq!(decision.enabled, vector.enabled);
            assert_eq!(
                serde_json::to_value(decision.reason).unwrap(),
                vector.decision
            );
        }
    }
}
