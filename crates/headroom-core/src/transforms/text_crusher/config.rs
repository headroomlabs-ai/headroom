//! TextCrusher configuration (Phase 2, #1171).
//!
//! Mirrors the Python `TextCrusherConfig`. Weights and thresholds are tuning
//! knobs for the recency + relevance + salience scoring.

#[derive(Debug, Clone)]
pub struct TextCrusherConfig {
    /// Keep roughly this fraction of characters.
    pub target_ratio: f64,
    pub w_recency: f64,
    pub w_relevance: f64,
    pub w_salience: f64,
    /// Segments shorter than this are de-prioritized (× 0.25).
    pub min_segment_chars: usize,
    /// Skip a candidate when this fraction of its word-shingles is already
    /// covered by kept segments (near-duplicate suppression).
    pub near_dup_threshold: f64,
    /// Below this many segments, pass through unchanged (nothing to gain).
    pub min_segments_for_crush: usize,
}

impl Default for TextCrusherConfig {
    fn default() -> Self {
        TextCrusherConfig {
            target_ratio: 0.5,
            w_recency: 1.0,
            w_relevance: 2.0,
            w_salience: 1.5,
            min_segment_chars: 12,
            near_dup_threshold: 0.85,
            min_segments_for_crush: 6,
        }
    }
}
