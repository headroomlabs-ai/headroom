//! CCR-backed extractive compression for prose leaves in structured payloads.

use crate::ccr::{compute_key, marker_for, CcrStore};
use crate::transforms::content_detector::detect_content_type;
use crate::transforms::pipeline::config::ProseFieldConfig;
use crate::transforms::pipeline::traits::{
    CompressionContext, OffloadOutput, OffloadTransform, TransformError,
};
use crate::transforms::text_crusher::TextCrusher;
use crate::transforms::ContentType;

const NAME: &str = "prose_field_offload";
const CONFIDENCE: f32 = 0.8;

pub struct ProseFieldOffload {
    crusher: TextCrusher,
    config: ProseFieldConfig,
}

impl ProseFieldOffload {
    pub fn new(config: ProseFieldConfig) -> Self {
        Self {
            crusher: TextCrusher::default(),
            config,
        }
    }

    pub fn config(&self) -> ProseFieldConfig {
        self.config
    }

    fn eligible(&self, content: &str) -> bool {
        content.len() >= self.config.min_bytes
            && detect_content_type(content).content_type == ContentType::PlainText
    }

    fn compress(&self, content: &str, query: &str) -> Option<(String, String)> {
        if !self.eligible(content) {
            return None;
        }
        let result = self
            .crusher
            .compress(content, query, Some(self.config.target_ratio));
        if result.total_segments < self.config.min_segments || result.compressed == content {
            return None;
        }

        let key = compute_key(content.as_bytes());
        let output = format!("{}\n{}", result.compressed, marker_for(&key));
        (output.len() < content.len()).then_some((output, key))
    }
}

impl OffloadTransform for ProseFieldOffload {
    fn name(&self) -> &'static str {
        NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &[ContentType::PlainText]
    }

    fn estimate_bloat(&self, content: &str) -> f32 {
        if !self.eligible(content) {
            return 0.0;
        }
        let segments = content
            .split(['.', '!', '?', '\n'])
            .filter(|segment| !segment.trim().is_empty())
            .count();
        if segments < self.config.min_segments {
            0.0
        } else {
            1.0
        }
    }

    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> Result<OffloadOutput, TransformError> {
        let Some((output, key)) = self.compress(content, &ctx.query) else {
            return Err(TransformError::skipped(
                NAME,
                "prose compression not worth it",
            ));
        };
        store.put(&key, content);
        Ok(OffloadOutput::from_lengths(content.len(), output, key))
    }

    fn confidence(&self) -> f32 {
        CONFIDENCE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::transforms::pipeline::config::PipelineConfig;

    fn offload() -> ProseFieldOffload {
        ProseFieldOffload::new(PipelineConfig::default().offload.prose_field)
    }

    fn prose() -> String {
        (0..12)
            .map(|i| {
                if i % 3 == 0 {
                    format!("Segment {i} documents recovery safeguards for this field.")
                } else {
                    format!("Segment {i} explains general context without the key term present.")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn short_plain_text_is_byte_identical() {
        let input = "A short note.";
        let store = InMemoryCcrStore::new();
        let result = offload().apply(input, &CompressionContext::default(), &store);
        assert!(result.is_err());
        assert!(store.is_empty());
    }

    #[test]
    fn long_low_segment_text_is_byte_identical() {
        let input = "A".repeat(300);
        let store = InMemoryCcrStore::new();
        let result = offload().apply(&input, &CompressionContext::default(), &store);
        assert!(result.is_err());
        assert!(store.is_empty());
    }

    #[test]
    fn estimate_bloat_is_zero_for_short_or_non_plain_text() {
        assert_eq!(offload().estimate_bloat("A short note."), 0.0);
        let html = "<html><body><p>".to_string() + &"x".repeat(300) + "</p></body></html>";
        assert_eq!(offload().estimate_bloat(&html), 0.0);
    }

    #[test]
    fn estimate_bloat_is_one_for_eligible_plain_text() {
        assert_eq!(offload().estimate_bloat(&prose()), 1.0);
    }

    #[test]
    fn estimate_bloat_is_zero_for_long_low_segment_plain_text() {
        let input = "A".repeat(300);
        assert_eq!(offload().estimate_bloat(&input), 0.0);
    }

    #[test]
    fn query_changes_selection_deterministically() {
        let input = prose();
        let crusher = offload();
        let a = crusher.compress(&input, "recovery").unwrap();
        let b = crusher.compress(&input, "recovery").unwrap();
        assert_eq!(a, b);
        assert!(a.0.contains("recovery"));
        assert!(!a
            .0
            .contains("Segment 1 explains general context without the key term present."));
    }
}
