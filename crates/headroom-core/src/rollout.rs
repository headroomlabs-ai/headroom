//! Release-channel and feature-flag policy shared by Rust components.
//!
//! Runtime feature gates should be resolved in composition roots. Lower-level
//! modules receive concrete collaborators or config values, not ad-hoc env
//! reads, so experiments cannot spread conditional logic through hot paths.

use std::collections::BTreeSet;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReleaseChannel {
    #[default]
    Stable,
    Beta,
    Canary,
    Dev,
}

impl ReleaseChannel {
    pub fn as_str(self) -> &'static str {
        match self {
            ReleaseChannel::Stable => "stable",
            ReleaseChannel::Beta => "beta",
            ReleaseChannel::Canary => "canary",
            ReleaseChannel::Dev => "dev",
        }
    }

    pub fn allows(self, required: ReleaseChannel) -> bool {
        self >= required
    }
}

impl FromStr for ReleaseChannel {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    NativeBedrock,
    OpenAiResponsesStreaming,
    CanaryProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSpec {
    pub name: &'static str,
    pub available_in: ReleaseChannel,
    pub default_enabled_in: Option<ReleaseChannel>,
}

impl Feature {
    pub fn spec(self) -> FeatureSpec {
        match self {
            Feature::NativeBedrock => FeatureSpec {
                name: "native_bedrock",
                available_in: ReleaseChannel::Stable,
                default_enabled_in: Some(ReleaseChannel::Stable),
            },
            Feature::OpenAiResponsesStreaming => FeatureSpec {
                name: "openai_responses_streaming",
                available_in: ReleaseChannel::Stable,
                default_enabled_in: Some(ReleaseChannel::Stable),
            },
            Feature::CanaryProbe => FeatureSpec {
                name: "canary_probe",
                available_in: ReleaseChannel::Canary,
                default_enabled_in: None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rollout {
    pub channel: ReleaseChannel,
    pub enabled: BTreeSet<String>,
    pub disabled: BTreeSet<String>,
    pub unsafe_allow_unstable: bool,
}

impl Default for Rollout {
    fn default() -> Self {
        Self {
            channel: ReleaseChannel::Stable,
            enabled: BTreeSet::new(),
            disabled: BTreeSet::new(),
            unsafe_allow_unstable: false,
        }
    }
}

impl Rollout {
    pub fn new(
        channel: ReleaseChannel,
        enabled: impl IntoIterator<Item = String>,
        disabled: impl IntoIterator<Item = String>,
        unsafe_allow_unstable: bool,
    ) -> Self {
        Self {
            channel,
            enabled: enabled.into_iter().map(normalize_feature_name).collect(),
            disabled: disabled.into_iter().map(normalize_feature_name).collect(),
            unsafe_allow_unstable,
        }
    }

    pub fn from_parts(
        channel: &str,
        enabled: &str,
        disabled: &str,
        unsafe_allow_unstable: bool,
    ) -> Self {
        Self::new(
            ReleaseChannel::from_str(channel).unwrap_or_default(),
            split_feature_names(enabled),
            split_feature_names(disabled),
            unsafe_allow_unstable,
        )
    }

    pub fn is_available(&self, feature: Feature) -> bool {
        let spec = feature.spec();
        self.channel.allows(spec.available_in) || self.unsafe_allow_unstable
    }

    pub fn is_enabled(&self, feature: Feature, explicit: bool) -> bool {
        let spec = feature.spec();
        if self.disabled.contains(spec.name) {
            return false;
        }
        if !self.is_available(feature) {
            return false;
        }
        if self.enabled.contains(spec.name) || explicit {
            return true;
        }
        spec.default_enabled_in
            .map(|channel| self.channel.allows(channel))
            .unwrap_or(false)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_channel_blocks_canary_feature_even_when_requested() {
        let rollout = Rollout::from_parts("stable", "canary_probe", "", false);

        assert!(!rollout.is_enabled(Feature::CanaryProbe, false));
        assert!(!rollout.is_enabled(Feature::CanaryProbe, true));
    }

    #[test]
    fn canary_channel_allows_explicit_canary_feature() {
        let rollout = Rollout::from_parts("canary", "", "", false);

        assert!(rollout.is_enabled(Feature::CanaryProbe, true));
    }

    #[test]
    fn unsafe_override_allows_unstable_feature_for_break_glass_only() {
        let rollout = Rollout::from_parts("stable", "canary_probe", "", true);

        assert!(rollout.is_enabled(Feature::CanaryProbe, false));
    }

    #[test]
    fn disable_list_wins_over_defaults() {
        let rollout = Rollout::from_parts("stable", "", "native-bedrock", false);

        assert!(!rollout.is_enabled(Feature::NativeBedrock, false));
    }
}
