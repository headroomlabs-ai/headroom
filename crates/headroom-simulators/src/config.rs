use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SimulatorConfig {
    pub stubs: Vec<StubRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StubRule {
    pub name: String,
    #[serde(default)]
    pub method: Option<String>,
    pub path: String,
    #[serde(default)]
    pub body_contains: Option<String>,
    #[serde(default)]
    pub body_json_pointer: Option<JsonPointerMatch>,
    pub response: ConfiguredResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPointerMatch {
    pub pointer: String,
    pub equals: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfiguredResponse {
    #[serde(default = "default_status")]
    pub status: u16,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub json: Option<Value>,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub sse: Vec<SseFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseFrame {
    #[serde(default)]
    pub event: Option<String>,
    pub data: Value,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read simulator config {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("simulator config is not valid JSON: {0}")]
    Parse(#[from] serde_json::Error),
}

pub fn load_config(path: Option<&Path>) -> Result<SimulatorConfig, ConfigError> {
    let Some(path) = path else {
        return Ok(SimulatorConfig::default());
    };
    let raw = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.display().to_string(),
        source,
    })?;
    Ok(serde_json::from_str(&raw)?)
}

fn default_status() -> u16 {
    200
}
