//! TextCrusher: fast deterministic extractive prose compressor (Phase 2, #1171).
//!
//! Splits prose into sentence segments, scores each by recency + query
//! relevance + structural salience, suppresses near-duplicates via a global
//! word-shingle index, and keeps the top segments (in original order) up to a
//! target ratio. Output is extractive: the kept sentences are verbatim words
//! (each segment trimmed, re-joined with `\n`) -- no invented words, no rewrite.
//!
//! The relevance term REUSES the shared [`BM25Scorer`](crate::relevance) for
//! non-CJK text. CJK has no spaces/ASCII terminators, so CJK-bearing input takes
//! an ICU (UAX#29 + dictionary) sentence/word segmentation path with a local
//! BM25 over the ICU tokens; pure-ASCII text is byte-identical to before.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::config::TextCrusherConfig;
use crate::relevance::{BM25Scorer, RelevanceScorer};
use icu_segmenter::{
    SentenceSegmenter, SentenceSegmenterBorrowed, WordSegmenter, WordSegmenterBorrowed,
};

// ICU segmenters resolved ONCE and reused (compiled_data is static, so the
// borrowed view is 'static). A fresh segmenter per call would dominate this
// compressor's request-path budget -- compress() tokenizes every segment.
static SENTENCE_SEGMENTER: LazyLock<SentenceSegmenterBorrowed<'static>> =
    LazyLock::new(|| SentenceSegmenter::new(Default::default()));
static WORD_SEGMENTER: LazyLock<WordSegmenterBorrowed<'static>> =
    LazyLock::new(|| WordSegmenter::new_dictionary(Default::default()));

/// True for CJK ideographs, kana, Hangul, plus CJK punctuation (。、「」) and
/// half/full-width forms — scripts/marks without ASCII spaces or terminators,
/// which the default ASCII splitter/tokenizer can't segment. CJK-bearing text
/// takes the ICU path; pure-ASCII text is byte-identical to before.
fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x3000..=0x303F        // CJK symbols & punctuation (。、「」【】)
            | 0x3040..=0x30FF  // Hiragana + Katakana
            | 0x3400..=0x4DBF  // CJK Ext A
            | 0x4E00..=0x9FFF  // CJK Unified
            | 0xAC00..=0xD7AF  // Hangul syllables
            | 0xF900..=0xFAFF  // CJK Compatibility ideographs
            | 0xFF00..=0xFFEF  // half/full-width forms (！？ ｶﾅ)
            | 0x20000..=0x2FA1F // CJK Ext B–F + Compat Supplement
    )
}

/// Token count for the reported ratio: ASCII whitespace words for non-CJK
/// (unchanged), CJK-aware tokens when CJK is present. Whitespace-splitting would
/// count a space-free CJK string as ONE token, making compression_ratio nonsense
/// (a newline-joined N-segment output looks like N tokens vs a 1-token input).
fn count_tokens(s: &str) -> usize {
    if s.chars().any(is_cjk) {
        tokens(s).len()
    } else {
        s.split_whitespace().count()
    }
}

const KEYWORDS: [&str; 10] = [
    "error",
    "exception",
    "failed",
    "failure",
    "fail",
    "warning",
    "traceback",
    "assert",
    "todo",
    "fixme",
];

#[derive(Debug, Clone)]
pub struct TextCrusherResult {
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub compression_ratio: f64,
    pub kept_segments: usize,
    pub total_segments: usize,
}

pub struct TextCrusher {
    config: TextCrusherConfig,
    scorer: BM25Scorer,
}

impl Default for TextCrusher {
    fn default() -> Self {
        TextCrusher::new(TextCrusherConfig::default())
    }
}

impl TextCrusher {
    pub fn new(config: TextCrusherConfig) -> Self {
        TextCrusher {
            config,
            scorer: BM25Scorer::default(),
        }
    }

    fn passthrough(content: &str, n_segments: usize) -> TextCrusherResult {
        let toks = count_tokens(content);
        TextCrusherResult {
            compressed: content.to_string(),
            original_tokens: toks,
            compressed_tokens: toks,
            compression_ratio: 1.0,
            kept_segments: n_segments,
            total_segments: n_segments,
        }
    }

    pub fn compress(
        &self,
        content: &str,
        context: &str,
        target_ratio: Option<f64>,
    ) -> TextCrusherResult {
        let cfg = &self.config;
        let ratio = target_ratio.unwrap_or(cfg.target_ratio).clamp(0.05, 1.0);

        let segments = split_segments(content);
        if segments.len() < cfg.min_segments_for_crush {
            return Self::passthrough(content, segments.len());
        }

        let n = segments.len();
        let total_chars: usize = segments.iter().map(|s| s.len()).sum();
        // .max(1) so a tiny input never truncates the budget to 0 (which would
        // admit nothing and silently fall back to a 100% passthrough).
        let target_chars = ((total_chars as f64 * ratio) as usize).max(1);

        let seg_tokens: Vec<Vec<String>> = segments.iter().map(|s| tokens(s)).collect();

        // CJK content: relevance via a local BM25 over the ICU word tokens (the
        // shared ASCII BM25Scorer scores zero terms for CJK). Dispatch on the
        // CONTENT only -- pure-ASCII content keeps the shared scorer even when
        // the query is CJK, so English output stays byte-identical.
        let relevance: Vec<f64> = if segments.iter().any(|s| s.chars().any(is_cjk)) {
            relevance_cjk(&seg_tokens, context)
        } else {
            let seg_refs: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();
            self.scorer
                .score_batch(&seg_refs, context)
                .iter()
                .map(|r| r.score)
                .collect()
        };

        let mut scores = vec![0.0f64; n];
        for i in 0..n {
            let recency = (i as f64 + 1.0) / n as f64;
            let rel = relevance.get(i).copied().unwrap_or(0.0);
            // CJK segments have no spaces, so split_whitespace yields one giant
            // "word" and zero salience; use the already-computed ICU tokens.
            let (salient, word_count) = if segments[i].chars().any(is_cjk) {
                let s = seg_tokens[i].iter().filter(|w| is_salient(w)).count();
                (s, seg_tokens[i].len())
            } else {
                let words: Vec<&str> = segments[i].split_whitespace().collect();
                (words.iter().filter(|w| is_salient(w)).count(), words.len())
            };
            let salience = salient as f64 / (word_count as f64 + 1.0);
            let mut score =
                cfg.w_recency * recency + cfg.w_relevance * rel + cfg.w_salience * salience;
            if segments[i].len() < cfg.min_segment_chars {
                score *= 0.25;
            }
            scores[i] = score;
        }

        // Highest score first; stable tiebreak by index for determinism.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut kept = vec![false; n];
        let mut seen: HashSet<String> = HashSet::new();
        let mut kept_chars = 0usize;
        let mut kept_count = 0usize;
        for &i in &order {
            if kept_chars >= target_chars {
                break;
            }
            let sh = shingles(&seg_tokens[i], 3);
            if !sh.is_empty() {
                let covered =
                    sh.iter().filter(|s| seen.contains(*s)).count() as f64 / sh.len() as f64;
                if covered >= cfg.near_dup_threshold {
                    continue; // near-duplicate: most shingles already kept
                }
            }
            kept[i] = true;
            kept_count += 1;
            for s in sh {
                seen.insert(s);
            }
            kept_chars += segments[i].len();
        }

        if kept_count == 0 {
            return Self::passthrough(content, n);
        }

        let compressed = (0..n)
            .filter(|&i| kept[i])
            .map(|i| segments[i].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let orig_tok = count_tokens(content);
        let comp_tok = count_tokens(&compressed);
        TextCrusherResult {
            compression_ratio: if orig_tok > 0 {
                comp_tok as f64 / orig_tok as f64
            } else {
                1.0
            },
            compressed,
            original_tokens: orig_tok,
            compressed_tokens: comp_tok,
            kept_segments: kept_count,
            total_segments: n,
        }
    }
}

/// Sentence/line segmentation, dispatched on content: ICU for CJK-bearing text,
/// the original ASCII splitter otherwise (byte-identical to before).
fn split_segments(text: &str) -> Vec<String> {
    if text.chars().any(is_cjk) {
        split_segments_icu(text)
    } else {
        split_segments_ascii(text)
    }
}

/// ASCII path (unchanged): on newlines, and after `.`/`!`/`?` + whitespace.
fn split_segments_ascii(text: &str) -> Vec<String> {
    let mut segs = Vec::new();
    for line in text.split('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut cur = String::new();
        let mut prev_term = false;
        for c in trimmed.chars() {
            if prev_term && c.is_whitespace() {
                let s = cur.trim();
                if !s.is_empty() {
                    segs.push(s.to_string());
                }
                cur.clear();
                prev_term = false;
                continue;
            }
            cur.push(c);
            prev_term = matches!(c, '.' | '!' | '?');
        }
        let s = cur.trim();
        if !s.is_empty() {
            segs.push(s.to_string());
        }
    }
    segs
}

/// CJK path: ICU/UAX#29 sentence boundaries per line, then the length fallback.
fn split_segments_icu(text: &str) -> Vec<String> {
    let seg = *SENTENCE_SEGMENTER;
    let mut out = Vec::new();
    for line in text.split('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut prev = 0usize;
        for b in seg.segment_str(trimmed) {
            if b > prev {
                let s = trimmed[prev..b].trim();
                if !s.is_empty() {
                    out.push(s.to_string());
                }
                prev = b;
            }
        }
        if prev < trimmed.len() {
            let s = trimmed[prev..].trim();
            if !s.is_empty() {
                out.push(s.to_string());
            }
        }
    }
    apply_length_fallback(out)
}

/// Mandatory terminator-sparse fallback: split any over-long CJK-bearing segment
/// on whitespace / CJK secondary punctuation, then a hard char cap.
fn apply_length_fallback(segs: Vec<String>) -> Vec<String> {
    let cap = 60usize;
    let hard = 40usize;
    let mut out = Vec::new();
    for s in segs {
        if s.chars().count() <= cap || !s.chars().any(is_cjk) {
            out.push(s);
            continue;
        }
        let mut piece = String::new();
        for c in s.chars() {
            piece.push(c);
            let n = piece.chars().count();
            let soft = c.is_whitespace() || matches!(c, '、' | '，' | '；' | '：' | '·' | '…');
            if (soft && n >= hard / 2) || n >= hard {
                let t = piece.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
                piece.clear();
            }
        }
        let t = piece.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }
    out
}

/// Word-unit tokenization for shingles/relevance, dispatched on content: ICU
/// word segmentation for CJK, the original ASCII alnum-run tokenizer otherwise.
fn tokens(text: &str) -> Vec<String> {
    if text.chars().any(is_cjk) {
        tokens_icu(text)
    } else {
        tokens_ascii(text)
    }
}

/// ASCII path (unchanged): lowercased alphanumeric/underscore runs.
fn tokens_ascii(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            for lc in c.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// CJK path: ICU WordSegmenter (dictionary) word units; alnum-bearing, lowercased.
fn tokens_icu(text: &str) -> Vec<String> {
    let seg = *WORD_SEGMENTER;
    let mut out = Vec::new();
    let mut prev = 0usize;
    for b in seg.segment_str(text) {
        if b > prev {
            let w = text[prev..b].trim();
            if !w.is_empty() && w.chars().any(|c| c.is_alphanumeric()) {
                out.push(w.to_lowercase());
            }
            prev = b;
        }
    }
    out
}

/// BM25 relevance over the segments' ICU word tokens. This is INTENTIONALLY a
/// separate scorer from the shared [`BM25Scorer`](crate::relevance): that one is
/// parity-locked to Python and tokenizes with an ASCII-only regex, so it scores
/// zero terms for CJK and cannot be reused here. This variant takes pre-computed
/// ICU word tokens and uses textbook BM25 (k1=1.2, b=0.75) -- deliberately NOT
/// BM25Scorer's ASCII-tuned k1=1.5 + long-identifier bonus, which don't transfer
/// to CJK words. The `+1` inside the idf log keeps it non-negative; output is
/// max-normalized to [0, 1] to match the shared scorer's range in the weighting.
fn relevance_cjk(seg_tokens: &[Vec<String>], context: &str) -> Vec<f64> {
    use std::collections::HashMap;
    let n = seg_tokens.len();
    let qtokens: HashSet<String> = tokens(context).into_iter().collect();
    if n == 0 || qtokens.is_empty() {
        return vec![0.0; n];
    }
    let mut df: HashMap<&str, usize> = HashMap::new();
    for toks in seg_tokens {
        let uniq: HashSet<&str> = toks.iter().map(|s| s.as_str()).collect();
        for t in uniq {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let nf = n as f64;
    let idf = |t: &str| -> f64 {
        let d = *df.get(t).unwrap_or(&0) as f64;
        (((nf - d + 0.5) / (d + 0.5)) + 1.0).ln()
    };
    let (k1, b) = (1.2_f64, 0.75_f64);
    let avgdl = (seg_tokens.iter().map(|t| t.len()).sum::<usize>() as f64 / nf).max(1.0);
    let mut out = vec![0.0f64; n];
    for (i, toks) in seg_tokens.iter().enumerate() {
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in toks {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        let dl = toks.len() as f64;
        let mut score = 0.0;
        for q in &qtokens {
            if let Some(&f) = tf.get(q.as_str()) {
                let f = f as f64;
                score += idf(q) * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl));
            }
        }
        out[i] = score;
    }
    let max = out.iter().cloned().fold(0.0f64, f64::max);
    if max > 0.0 {
        for s in &mut out {
            *s /= max;
        }
    }
    out
}

fn shingles(words: &[String], k: usize) -> HashSet<String> {
    let mut set = HashSet::new();
    if words.is_empty() {
        return set;
    }
    if words.len() < k {
        // Short segment: emit every sub-window (1..=len) so identical/overlapping
        // short segments still near-dup-match each other. (They can't match a
        // longer segment's k-grams, but short segments are score-penalized and
        // rarely survive selection anyway.)
        for size in 1..=words.len() {
            for w in words.windows(size) {
                set.insert(w.join("\u{1}"));
            }
        }
        return set;
    }
    for w in words.windows(k) {
        set.insert(w.join("\u{1}"));
    }
    set
}

/// A word carries specific, hard-to-reconstruct information if it has a digit,
/// is an error/status keyword, is ALLCAPS (2+ letters), or is a dotted
/// identifier (`foo.bar`).
fn is_salient(word: &str) -> bool {
    if word.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    let lower = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if KEYWORDS.contains(&lower.as_str()) {
        return true;
    }
    let alpha: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    if alpha.len() >= 2 && alpha.iter().all(|c| c.is_uppercase()) {
        return true;
    }
    if let Some(dot) = word.find('.') {
        let a = &word[..dot];
        let b = &word[dot + 1..];
        if !a.is_empty()
            && !b.is_empty()
            && a.chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
            && b.chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(n: usize) -> String {
        (0..n)
            .map(|i| format!("Sentence number {i} describes a distinct topic {i} in some detail."))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn extractive_and_compresses() {
        let content = doc(40);
        let r = TextCrusher::default().compress(&content, "", Some(0.3));
        assert!(r.compressed_tokens < r.original_tokens);
        // extractive: every output word appears in the input
        let orig: HashSet<&str> = content.split_whitespace().collect();
        assert!(r.compressed.split_whitespace().all(|w| orig.contains(w)));
    }

    #[test]
    fn deterministic() {
        let content = doc(40);
        let tc = TextCrusher::default();
        assert_eq!(
            tc.compress(&content, "", Some(0.4)).compressed,
            tc.compress(&content, "", Some(0.4)).compressed
        );
    }

    #[test]
    fn passthrough_when_small() {
        let r = TextCrusher::default().compress("one. two. three.", "", None);
        assert_eq!(r.compression_ratio, 1.0);
    }

    #[test]
    fn cjk_splits_on_full_width_terminators() {
        let zh = "今天天气很好。我们去公园散步。然后回家吃饭。下午还要开会。晚上看电影。";
        let segs = split_segments(zh);
        assert!(
            segs.len() >= 4,
            "expected multiple CJK sentences, got {segs:?}"
        );
        for s in &segs {
            assert!(zh.contains(s.as_str()), "segment not verbatim: {s}");
        }
    }

    #[test]
    fn cjk_terminator_sparse_length_fallback() {
        // a long flowing CJK run with NO terminators must STILL split (the fallback)
        let zh = "机器学习模型从数据中学习特征并识别模式进行预测的系统会不断地调整参数\
                  以最小化误差从而提升准确率这是一段很长的没有任何标点的中文用来测试兜底\
                  切分是否生效以及能否产生多个段落供后续打分与去重使用确保不会整段透传";
        assert!(
            split_segments(zh).len() >= 2,
            "terminator-sparse CJK must still split into multiple segments"
        );
    }

    #[test]
    fn cjk_tokens_not_one_giant_token() {
        // the old ASCII tokenizer collapsed a whole Han run into ONE token
        assert!(
            tokens("数据库连接失败重试三次").len() >= 3,
            "CJK run must yield multiple word-ish tokens"
        );
    }

    #[test]
    fn cjk_relevance_keeps_query_match() {
        let needle = "认证令牌的缓存策略采用最近最少使用淘汰算法来管理过期。";
        let filler = "今天天气很好。我们去公园散步。然后回家吃饭。下午还要开会。\
                      晚上看电影。明天继续工作。周末去爬山。后天有个会议。";
        let doc = format!("{filler}{needle}{filler}");
        let r = TextCrusher::default().compress(&doc, "认证令牌缓存策略", Some(0.3));
        assert!(r.compressed_tokens < r.original_tokens, "should compress");
        assert!(
            r.compressed.contains("认证令牌"),
            "query-relevant CJK sentence must survive selection: {}",
            r.compressed
        );
    }

    #[test]
    fn mixed_cjk_latin_keeps_ascii_terms_and_compresses() {
        let content = "系统启动失败。认证模块超时。\nERROR: connection refused at host.\n\
                       数据库连接池耗尽。重试机制触发。服务降级处理完成。请检查日志。";
        let r = TextCrusher::default().compress(content, "ERROR connection", Some(0.5));
        assert!(r.compression_ratio < 1.0, "mixed content must compress");
        assert!(r.original_tokens > 5, "must not collapse CJK to one token");
        assert!(
            r.compressed.contains("ERROR"),
            "ASCII term relevant to the query must survive: {}",
            r.compressed
        );
    }

    #[test]
    fn korean_tokenizes_and_splits() {
        let ko = "인증 토큰의 캐시 전략은 최근 최소 사용 알고리즘으로 관리된다。\
                  세션은 만료 시간에 따라 자동으로 정리된다。";
        assert!(
            tokens(ko).len() >= 4,
            "Korean must yield multiple tokens via ICU"
        );
        assert!(split_segments(ko).len() >= 2, "Korean sentences must split");
    }

    #[test]
    fn japanese_no_space_tokenizes_via_dictionary() {
        // Japanese has no spaces; ICU dictionary segmentation must still split
        // this into multiple word tokens (whitespace-splitting would give one).
        let ja = "認証トークンのキャッシュ戦略は最近最少使用アルゴリズムで管理される。\
                  セッションは有効期限に従って自動的に整理される。";
        assert!(
            tokens(ja).len() >= 5,
            "Japanese must split into multiple ICU tokens"
        );
        assert!(
            split_segments(ja).len() >= 2,
            "Japanese sentences must split on 。"
        );
    }

    #[test]
    fn cjk_token_count_and_ratio_are_sane() {
        let zh = "系统架构遵循微服务模式。每个服务拥有自己的数据存储和接口。".repeat(20);
        let r = TextCrusher::default().compress(&zh, "", Some(0.4));
        assert!(
            r.original_tokens > 10,
            "CJK token count must not collapse to 1"
        );
        assert!(r.compressed_tokens < r.original_tokens);
        assert!(r.compression_ratio > 0.0 && r.compression_ratio <= 1.0);
    }

    #[test]
    fn ascii_content_unchanged_even_with_cjk_query() {
        // C-fix: dispatch is on CONTENT, not query. A CJK query against pure-ASCII
        // content must take the unchanged ASCII path (shared BM25Scorer).
        let content = doc(40);
        let with_ascii_q = TextCrusher::default().compress(&content, "topic 7", Some(0.3));
        let with_cjk_q = TextCrusher::default().compress(&content, "主题 七", Some(0.3));
        // ASCII content compresses identically regardless of the query script
        assert_eq!(with_ascii_q.total_segments, with_cjk_q.total_segments);
        assert!(with_cjk_q.compressed_tokens < with_cjk_q.original_tokens);
    }
}
