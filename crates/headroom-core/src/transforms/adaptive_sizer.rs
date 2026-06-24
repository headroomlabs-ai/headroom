//! Adaptive compression sizing via information saturation detection.
//!
//! Direct port of `headroom/transforms/adaptive_sizer.py`. Used by
//! `smart_crusher`'s array crushers to decide *how many* items to keep —
//! statistically, by detecting the "knee point" of an information
//! saturation curve.
//!
//! # Algorithm overview
//!
//! Three-tier decision:
//! 1. **Fast path**: trivial cases (`n <= 8` → keep all) and near-total
//!    redundancy (≤3 unique-by-simhash → keep that count).
//! 2. **Standard**: Kneedle on cumulative unique-bigram coverage curve.
//!    Coverage stops growing → that's the knee → return that count.
//! 3. **Validation**: zlib-ratio sanity check. If keeping `k` items
//!    produces a much-more-redundant subset than the full set, bump
//!    `k` by 20%.
//!
//! # Parity-relevant subtleties
//!
//! - `_simhash` hashes character 4-grams via MD5, then aggregates bits
//!   via weighted voting. Take the first 64 bits of the MD5 digest as a
//!   big-endian `u64` — Python does `int(hex[:16], 16)` which is exactly
//!   that. Per-character iteration matches Python's `str` slicing
//!   (codepoints, not bytes).
//! - `compute_unique_bigram_curve` operates on whitespace-split words.
//!   Single-word items emit `(word, "")`. Empty-string items emit
//!   `("", "")`. Both languages must agree byte-for-byte on the set
//!   cardinality.
//! - `find_knee` requires `> 0.05` deviation from the diagonal in
//!   normalized space; threshold is strict (`<` returns None).
//! - `_validate_with_zlib` uses `zlib.compress(..., level=1)`. We use
//!   `flate2` with the default miniz_oxide backend; for typical inputs
//!   the output length matches CPython's libz at level=1 closely enough
//!   that the 15% ratio-diff threshold absorbs any per-byte drift.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use md5::{Digest, Md5};
use std::collections::HashSet;
use std::io::Write;

/// Compute the optimal number of items to keep via information saturation.
///
/// Direct port of `compute_optimal_k` (Python `adaptive_sizer.py:27-106`).
///
/// # Arguments
///
/// - `items`: string representations of items in importance order.
/// - `bias`: multiplier on the knee point (>1 = keep more, <1 = compress
///   harder).
/// - `min_k`: lower bound on the return value.
/// - `max_k`: upper bound; `None` means "no cap" (i.e. up to `items.len()`).
pub fn compute_optimal_k(items: &[&str], bias: f64, min_k: usize, max_k: Option<usize>) -> usize {
    let n = items.len();
    let effective_max = max_k.unwrap_or(n);

    // Tier 1: fast path.
    if n <= 8 {
        return n;
    }

    // Near-total redundancy: at most 3 unique groups → keep that many.
    let unique_count = count_unique_simhash(items, 3);
    if unique_count <= 3 {
        let k = min_k.max(unique_count);
        return k.min(effective_max);
    }

    // Tier 2: Kneedle on bigram-coverage curve.
    let curve = compute_unique_bigram_curve(items);
    let mut knee = find_knee(&curve);

    // Diversity ratio: fraction of items that are genuinely unique.
    let diversity_ratio = unique_count as f64 / n as f64;

    knee = match knee {
        None => {
            // No saturation found — scale keep-fraction with diversity.
            // diversity ~1.0 → keep 100%; ~0.0 → keep 30%.
            let keep_fraction = 0.3 + 0.7 * diversity_ratio;
            Some(min_k.max((n as f64 * keep_fraction) as usize))
        }
        Some(k) if diversity_ratio > 0.7 => {
            // Knee found, but high diversity — apply diversity floor so
            // we don't drop below `n * (0.3 + 0.7 * diversity)`.
            let floor = min_k.max((n as f64 * (0.3 + 0.7 * diversity_ratio)) as usize);
            Some(k.max(floor))
        }
        some => some,
    };

    let knee = knee.unwrap_or(min_k); // defensive — knee path always sets Some above

    // Apply bias multiplier. Python: `int(knee * bias)`.
    let mut k = min_k.max((knee as f64 * bias) as usize);
    k = k.min(effective_max);

    // Tier 3: zlib-ratio validation.
    k = validate_with_zlib(items, k, effective_max, 0.15);

    // Final clamp.
    min_k.max(k.min(effective_max))
}

/// Find the knee in a monotonically-increasing curve (Kneedle).
///
/// Direct port of `find_knee` (Python `adaptive_sizer.py:109-154`).
/// Returns the 1-indexed count `knee_idx + 1` so the caller can use it
/// directly as a "keep this many" value.
pub fn find_knee(curve: &[usize]) -> Option<usize> {
    let n = curve.len();
    if n < 3 {
        return None;
    }

    let x_min: usize = 0;
    let x_max: usize = n - 1;
    let y_min = curve[0] as f64;
    let y_max = curve[n - 1] as f64;

    if (y_max - y_min).abs() < f64::EPSILON {
        // Flat curve — all items are identical.
        // Python returns the literal `1`.
        return Some(1);
    }

    let x_range = (x_max - x_min) as f64;
    let y_range = y_max - y_min;

    let mut max_diff: f64 = -1.0;
    let mut knee_idx: Option<usize> = None;

    for (i, &y) in curve.iter().enumerate() {
        let x_norm = (i - x_min) as f64 / x_range;
        let y_norm = (y as f64 - y_min) / y_range;
        let diff = y_norm - x_norm;
        if diff > max_diff {
            max_diff = diff;
            knee_idx = Some(i);
        }
    }

    if max_diff < 0.05 {
        return None;
    }

    knee_idx.map(|i| i + 1)
}

/// Cumulative unique-bigram coverage curve.
///
/// Direct port of `compute_unique_bigram_curve` (Python
/// `adaptive_sizer.py:157-182`). Each item contributes its word-level
/// bigrams; single-word items contribute `(word, "")`. The curve at
/// index `k` is the running count of unique bigrams after seeing
/// `items[0..=k]`.
pub fn compute_unique_bigram_curve(items: &[&str]) -> Vec<usize> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut curve: Vec<usize> = Vec::with_capacity(items.len());

    for item in items {
        let lower = item.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        if words.len() < 2 {
            // Single word or empty: synthesize a unigram-bigram.
            let first = words.first().copied().unwrap_or("");
            seen.insert((first.to_string(), String::new()));
        } else {
            for j in 0..words.len() - 1 {
                seen.insert((words[j].to_string(), words[j + 1].to_string()));
            }
        }
        curve.push(seen.len());
    }

    curve
}

/// 64-bit SimHash fingerprint of a text string.
///
/// Direct port of `_simhash` (Python `adaptive_sizer.py:185-214`).
/// Algorithm:
/// 1. Iterate character 4-grams (sliding window). For input shorter
///    than 4 chars, the loop runs once with the entire string as the
///    only "gram". Empty input still iterates once with `""`.
/// 2. Hash each gram with MD5; take the first 64 bits as a big-endian
///    `u64`. (Python: `int(hexdigest()[:16], 16)`.)
/// 3. For each bit position 0..64, increment a vote counter when the
///    bit is set, decrement when clear.
/// 4. Final fingerprint: bit `j` is set iff `votes[j] > 0` (strict).
pub fn simhash(text: &str) -> u64 {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let n = chars.len();

    // Python: `range(max(1, len(text_lower) - 3))`. For n<=3, this is
    // `range(1)` (single iteration on the whole string). For n>=4 it's
    // `range(n-3)`.
    let iter_count = if n <= 3 { 1 } else { n - 3 };

    let mut votes: [i32; 64] = [0; 64];

    for i in 0..iter_count {
        // 4-character window starting at char index i. For short input,
        // this is just the whole string padded by being shorter than 4.
        let gram: String = chars.iter().skip(i).take(4).collect();

        let digest = Md5::digest(gram.as_bytes());
        // First 8 bytes of the 16-byte digest, big-endian → u64.
        // Mirrors Python's `int(hex[:16], 16)` exactly.
        let h = u64::from_be_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]);

        for (j, vote) in votes.iter_mut().enumerate() {
            if (h >> j) & 1 == 1 {
                *vote += 1;
            } else {
                *vote -= 1;
            }
        }
    }

    let mut fingerprint: u64 = 0;
    for (j, &v) in votes.iter().enumerate() {
        if v > 0 {
            fingerprint |= 1 << j;
        }
    }
    fingerprint
}

/// Hamming distance between two 64-bit SimHash fingerprints.
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Count items with distinct content via SimHash + greedy clustering.
///
/// Direct port of `count_unique_simhash` (Python `adaptive_sizer.py:222-252`).
/// Two items cluster together when their fingerprints are within
/// `threshold` Hamming distance.
pub fn count_unique_simhash(items: &[&str], threshold: u32) -> usize {
    if items.is_empty() {
        return 0;
    }

    let fingerprints: Vec<u64> = items.iter().map(|s| simhash(s)).collect();
    let mut clusters: Vec<u64> = Vec::new();

    for &fp in &fingerprints {
        let mut matched = false;
        for &rep in &clusters {
            if hamming_distance(fp, rep) <= threshold {
                matched = true;
                break;
            }
        }
        if !matched {
            clusters.push(fp);
        }
    }

    clusters.len()
}

/// zlib-based compression-ratio validation of the chosen `k`.
///
/// Direct port of `_validate_with_zlib` (Python `adaptive_sizer.py:255-308`).
/// If the subset `items[..k]` compresses *much* better than the full
/// set, the subset is missing diversity → bump `k` by 20%.
///
/// `tolerance` is the maximum allowed ratio difference (Python default
/// 0.15 = 15%).
pub fn validate_with_zlib(items: &[&str], k: usize, max_k: usize, tolerance: f64) -> usize {
    if k >= items.len() || k >= max_k {
        return k;
    }

    let full_text = items.join("\n");
    let subset_text = items[..k].join("\n");

    // Skip validation for very small content (zlib overhead dominates).
    if full_text.len() < 200 {
        return k;
    }

    let full_compressed = zlib_compressed_len(full_text.as_bytes());
    let subset_compressed = zlib_compressed_len(subset_text.as_bytes());

    let full_ratio = if !full_text.is_empty() {
        full_compressed as f64 / full_text.len() as f64
    } else {
        1.0
    };
    let subset_ratio = if !subset_text.is_empty() {
        subset_compressed as f64 / subset_text.len() as f64
    } else {
        1.0
    };

    let ratio_diff = (full_ratio - subset_ratio).abs();

    if ratio_diff > tolerance {
        // Subset compresses much better than full → bump k by 20%.
        let adjusted = ((k as f64) * 1.2) as usize;
        return adjusted.min(max_k);
    }

    k
}

/// Compress `bytes` with zlib level=1 and return the output length.
///
/// Wraps `flate2::ZlibEncoder` at `Compression::fast()` (level 1).
/// Mirrors Python's `len(zlib.compress(data, level=1))`. miniz_oxide
/// (default flate2 backend) produces DEFLATE streams of similar length
/// to CPython's libz at level 1 — small per-byte drift is absorbed by
/// the 15% ratio-diff tolerance in `validate_with_zlib`.
fn zlib_compressed_len(bytes: &[u8]) -> usize {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    // Writes are infallible for an in-memory Vec.
    encoder.write_all(bytes).expect("in-memory write");
    let compressed = encoder.finish().expect("flush");
    compressed.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- simhash (verified against Python reference) ----------

    #[test]
    fn simhash_empty_string() {
        // md5("")[:16] = "d41d8cd98f00b204"
        assert_eq!(simhash(""), 0xd41d8cd98f00b204);
    }

    #[test]
    fn simhash_single_char() {
        // md5("a")[:16] = "0cc175b9c0f1b6a8"
        assert_eq!(simhash("a"), 0x0cc175b9c0f1b6a8);
    }

    #[test]
    fn simhash_short_strings() {
        // For n <= 3, single iteration; fp = md5(text)[:16] as u64.
        assert_eq!(simhash("ab"), 0x187ef4436122d1cc);
        assert_eq!(simhash("abc"), 0x900150983cd24fb0);
    }

    #[test]
    fn simhash_n_eq_4_single_iteration() {
        // n=4: max(1, 4-3)=1, single iteration on full string.
        assert_eq!(simhash("abcd"), 0xe2fc714c4727ee93);
    }

    #[test]
    fn simhash_multi_window() {
        // n>=5 → bit voting from multiple grams.
        assert_eq!(simhash("hello"), 0x0209020130100020);
        assert_eq!(simhash("hello world"), 0x4681260120120222);
    }

    #[test]
    fn simhash_unicode_codepoint_iteration() {
        // "café" is 4 codepoints — should iterate once on the full string,
        // hash md5 of UTF-8 bytes (5 bytes for é=2 bytes).
        assert_eq!(simhash("café"), 0x07117fe4a1ebd544);
    }

    #[test]
    fn simhash_lowercases_input() {
        // Python lowercases before hashing.
        assert_eq!(simhash("ABC"), simhash("abc"));
        assert_eq!(simhash("Hello"), simhash("hello"));
    }

    #[test]
    fn simhash_longer_text() {
        assert_eq!(simhash("The quick brown fox jumps"), 0x30875e2639b3cb98);
    }

    // ---------- hamming_distance ----------

    #[test]
    fn hamming_distance_zero_identical() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0xff, 0xff), 0);
    }

    #[test]
    fn hamming_distance_basic() {
        assert_eq!(hamming_distance(0b0000, 0b1111), 4);
        assert_eq!(hamming_distance(0b1010, 0b0101), 4);
        assert_eq!(hamming_distance(0b1100, 0b1010), 2);
    }

    #[test]
    fn hamming_distance_full_64_bits() {
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }

    // ---------- count_unique_simhash ----------

    #[test]
    fn count_unique_simhash_empty() {
        assert_eq!(count_unique_simhash(&[], 3), 0);
    }

    #[test]
    fn count_unique_simhash_all_identical() {
        let items = ["abc", "abc", "abc"];
        assert_eq!(count_unique_simhash(&items, 3), 1);
    }

    #[test]
    fn count_unique_simhash_diverse_items() {
        // Three sentences with very different bigram coverage — should
        // simhash to fingerprints with Hamming > 3.
        let items = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a fish swam in the sea",
        ];
        assert_eq!(count_unique_simhash(&items, 3), 3);
    }

    #[test]
    fn count_unique_simhash_threshold_groups_near_dupes() {
        // Same fingerprint distance — well under threshold.
        let items = ["abc", "abc"];
        assert_eq!(count_unique_simhash(&items, 0), 1);
    }

    // ---------- compute_unique_bigram_curve ----------

    #[test]
    fn bigram_curve_distinct_words() {
        // ["the cat", "the dog", "a fish"] → [1, 2, 3]
        let items = ["the cat", "the dog", "a fish"];
        assert_eq!(compute_unique_bigram_curve(&items), vec![1, 2, 3]);
    }

    #[test]
    fn bigram_curve_single_word_dedup() {
        // ["hello", "world", "hello"] → [1, 2, 2]  (third "hello" dupes)
        let items = ["hello", "world", "hello"];
        assert_eq!(compute_unique_bigram_curve(&items), vec![1, 2, 2]);
    }

    #[test]
    fn bigram_curve_empty_string_contributes_one() {
        // ["", "a", "a b"] → [1, 2, 3]
        // "" → ("", "")
        // "a" → ("a", "")
        // "a b" → ("a", "b")
        let items = ["", "a", "a b"];
        assert_eq!(compute_unique_bigram_curve(&items), vec![1, 2, 3]);
    }

    #[test]
    fn bigram_curve_lowercases_for_dedup() {
        // "Hello" and "hello" should produce the same bigram.
        let items = ["Hello", "hello"];
        assert_eq!(compute_unique_bigram_curve(&items), vec![1, 1]);
    }

    // ---------- find_knee ----------

    #[test]
    fn find_knee_too_short_is_none() {
        assert_eq!(find_knee(&[]), None);
        assert_eq!(find_knee(&[1]), None);
        assert_eq!(find_knee(&[1, 2]), None);
    }

    #[test]
    fn find_knee_flat_curve_returns_one() {
        // y_max == y_min → return 1 (Python literal).
        assert_eq!(find_knee(&[5, 5, 5, 5, 5]), Some(1));
    }

    #[test]
    fn find_knee_concave_curve() {
        // Reference computed via Python: [1,5,8,9,10,10,10,10,10] → 3
        assert_eq!(find_knee(&[1, 5, 8, 9, 10, 10, 10, 10, 10]), Some(3));
    }

    #[test]
    fn find_knee_linear_no_clear_knee() {
        // Diagonal curve → max_diff = 0 < 0.05 → None.
        assert_eq!(find_knee(&[1, 2, 3, 4, 5, 6, 7, 8, 9]), None);
    }

    // ---------- validate_with_zlib ----------

    #[test]
    fn validate_zlib_passthrough_when_k_at_max() {
        // k >= len(items) → no adjustment.
        let items = ["a", "b", "c"];
        assert_eq!(validate_with_zlib(&items, 3, 10, 0.15), 3);
    }

    #[test]
    fn validate_zlib_passthrough_when_total_too_small() {
        // total bytes < 200 → skip validation (per Python).
        let items: [&str; 5] = ["short"; 5];
        assert_eq!(validate_with_zlib(&items, 2, 100, 0.15), 2);
    }

    #[test]
    fn validate_zlib_bumps_k_when_subset_undercompresses() {
        // Counterintuitive: 20 identical lines and 5 identical lines have
        // the same content redundancy, but zlib at level=1 compresses
        // longer redundant text more efficiently per byte. The validator
        // sees a ratio_diff > 0.15 between full and subset → bumps k by
        // 20%. Verified against Python: returns 6 for k=5.
        let items: [&str; 20] = ["the quick brown fox jumps over the lazy dog"; 20];
        let result = validate_with_zlib(&items, 5, 100, 0.15);
        assert_eq!(result, 6, "expected 1.2× bump from 5 to 6");
    }

    #[test]
    fn validate_zlib_passthrough_when_subset_representative() {
        // 20 diverse items with similar per-item compressibility — full
        // and subset get similar ratios → no bump.
        let many: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "entry id={} payload=item value with content for item number {}",
                    i, i
                )
            })
            .collect();
        let items: Vec<&str> = many.iter().map(|s| s.as_str()).collect();
        let result = validate_with_zlib(&items, 10, 100, 0.15);
        // With 10 of 20 diverse items, ratio_diff should stay under 0.15.
        // Pin to the equality observed; if zlib backend changes shift it,
        // we'll see a clean signal here.
        assert_eq!(result, 10, "expected passthrough for representative subset");
    }

    // ---------- compute_optimal_k (parity with Python) ----------

    #[test]
    fn compute_optimal_k_n_le_8_returns_n() {
        let items = ["a", "b", "c", "d", "e"];
        assert_eq!(compute_optimal_k(&items, 1.0, 3, None), 5);
    }

    #[test]
    fn compute_optimal_k_low_diversity_returns_unique_count() {
        // 10 identical → unique=1 → max(min_k=3, 1) = 3.
        let items: [&str; 10] = ["abc"; 10];
        assert_eq!(compute_optimal_k(&items, 1.0, 3, None), 3);
    }

    #[test]
    fn compute_optimal_k_all_unique_keeps_all() {
        // 20 distinct items, no knee, diversity_ratio=1.0 → keep ~100% → 20.
        let items: Vec<String> = (0..20)
            .map(|i| format!("unique item number {} with some long content", i))
            .collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        assert_eq!(compute_optimal_k(&refs, 1.0, 3, None), 20);
    }

    #[test]
    fn compute_optimal_k_respects_max_k() {
        let items: Vec<String> = (0..20).map(|i| format!("item {}", i)).collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let k = compute_optimal_k(&refs, 1.0, 3, Some(10));
        assert!(k <= 10, "k={} should be ≤ max_k=10", k);
    }

    #[test]
    fn compute_optimal_k_respects_min_k() {
        // Force a path that would return fewer than min_k by pinning
        // tons of identical items + high min_k.
        let items: [&str; 20] = ["abc"; 20];
        let k = compute_optimal_k(&items, 1.0, 5, None);
        assert_eq!(k, 5);
    }

    #[test]
    fn compute_optimal_k_bias_keeps_more() {
        // Higher bias should give >= the unbiased k.
        let items: Vec<String> = (0..30).map(|i| format!("item content {}", i)).collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let k_low = compute_optimal_k(&refs, 0.7, 3, None);
        let k_mid = compute_optimal_k(&refs, 1.0, 3, None);
        let k_high = compute_optimal_k(&refs, 1.5, 3, None);
        assert!(
            k_low <= k_mid,
            "bias 0.7 → {} should be ≤ bias 1.0 → {}",
            k_low,
            k_mid
        );
        assert!(
            k_mid <= k_high,
            "bias 1.0 → {} should be ≤ bias 1.5 → {}",
            k_mid,
            k_high
        );
    }
}
