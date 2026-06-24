//! Tag protection — keep custom workflow tags out of ML compressors.
//!
//! # Why this exists
//!
//! LLM workflows carry XML-style markers (`<system-reminder>`,
//! `<tool_call>`, `<thinking>`, `<headroom:tool_digest>`, etc.) that
//! downstream code parses as structure. Kompress / LLMLingua sees them
//! as droppable noise and silently strips them, breaking everything
//! that depends on them. ContentRouter calls [`protect_tags`] before
//! every ML-text-compression to swap custom-tag spans for opaque
//! placeholders, runs the compressor on the cleaned body, then calls
//! [`restore_tags`] on the output to splice the originals back in.
//!
//! Standard HTML5 elements (`<div>`, `<p>`, `<span>`, …) are *not*
//! protected — those go through the HTMLExtractor / trafilatura path
//! at a different layer. Anything else is treated as a custom tag.
//!
//! # Algorithm
//!
//! Single-pass tag-stack walker over the input bytes (no regex
//! backtracking, no O(n²) restart loop):
//!
//! 1. Scan forward for `<`. If the next bytes form a valid tag-open
//!    (`<name attr=…>` or `<name/>`), classify the tag name.
//! 2. HTML tag → emit verbatim, continue.
//! 3. Custom tag, self-closing → emit a placeholder, record the span.
//! 4. Custom tag, opening → push `(name, start_offset)` onto a stack.
//! 5. `</name>` matching the top of the stack → pop, emit a placeholder
//!    for the whole `<name>…</name>` span (when
//!    `compress_tagged_content == false`) or emit two placeholders for
//!    the markers only (when `compress_tagged_content == true`).
//! 6. Mismatched close (HTML close while a custom tag is on top, or a
//!    close with no matching open) → write the close tag verbatim and
//!    move on. The walker never attempts to "repair" malformed input.
//!
//! Output is built incrementally with offset-based slicing — never the
//! Python-original's `result.replace(original, placeholder, 1)`, which
//! silently misbehaves when two identical custom-tag blocks appear in
//! the same input (it always replaces the *first* textual occurrence,
//! not the matched one). See `fixed_in_3e4_replace_first` test below.
//!
//! # Bug fixes vs the Python original
//!
//! * **#1: O(n²) on nested custom tags** — the Python loop restarted a
//!   full regex scan after every replacement. Rust walks once, in
//!   linear time on input length.
//! * **#2: First-occurrence replace bug** — `str.replace(.., .., 1)`
//!   replaced the first textual match of the matched block, not the
//!   block at the matched offset. Two identical custom-tag blocks in
//!   the same input collapsed to one placeholder + a duplicated
//!   second block. The Rust walker stitches output by offset.
//! * **#3: Silent 50-iteration cap** — Python had a hard 50-iteration
//!   safety limit that quietly truncated tag protection on deeply
//!   nested input. The Rust walker's run-time is bounded by input
//!   length only.
//! * **#4: Self-closing pass duplicate-replace risk** — Python ran a
//!   second loop with the same `replace_first` bug for self-closing
//!   tags. Rust handles self-closers in the same single pass.
//! * **#5: Placeholder collision** — if input contains a literal
//!   `{{HEADROOM_TAG_…}}` substring, Python silently let the collision
//!   stand. We detect that and pick a salted prefix (with a tracing
//!   warn) so restoration can't be ambiguous.
//!
//! # Hot path
//!
//! `protect_tags` runs on every ML-compression call from ContentRouter.
//! Most production prompts have 0–10 custom tags so the absolute cost
//! is small either way; the value of the port is correctness (bugs
//! #2, #5) and predictable behavior on adversarial input (bugs #1,
//! #3). The PyO3 bridge releases the GIL during the walk because the
//! algorithm is fully self-contained.

use std::collections::HashSet;
use std::sync::OnceLock;

/// HTML5 living-standard element names — the set of tags this module
/// will NEVER protect (they're handled at a different layer; everything
/// else is treated as custom).
///
/// Generated from
/// <https://html.spec.whatwg.org/multipage/indices.html#elements-3> and
/// matches the Python `KNOWN_HTML_TAGS` frozenset element-for-element
/// so the Rust shim and the Python shim agree.
const HTML5_TAGS: &[&str] = &[
    // Main root
    "html",
    // Document metadata
    "base",
    "head",
    "link",
    "meta",
    "style",
    "title",
    // Sectioning root
    "body",
    // Content sectioning
    "address",
    "article",
    "aside",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hgroup",
    "main",
    "nav",
    "section",
    "search",
    // Text content
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "hr",
    "li",
    "menu",
    "ol",
    "p",
    "pre",
    "ul",
    // Inline text semantics
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "br",
    "cite",
    "code",
    "data",
    "dfn",
    "em",
    "i",
    "kbd",
    "mark",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
    "var",
    "wbr",
    // Image and multimedia
    "area",
    "audio",
    "img",
    "map",
    "track",
    "video",
    // Embedded content
    "embed",
    "iframe",
    "object",
    "param",
    "picture",
    "portal",
    "source",
    // SVG and MathML
    "svg",
    "math",
    // Scripting
    "canvas",
    "noscript",
    "script",
    // Demarcating edits
    "del",
    "ins",
    // Table content
    "caption",
    "col",
    "colgroup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    // Forms
    "button",
    "datalist",
    "fieldset",
    "form",
    "input",
    "label",
    "legend",
    "meter",
    "optgroup",
    "option",
    "output",
    "progress",
    "select",
    "textarea",
    // Interactive
    "details",
    "dialog",
    "summary",
    // Web Components
    "slot",
    "template",
];

fn known_html_tags() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| HTML5_TAGS.iter().copied().collect())
}

/// Default placeholder prefix. Brace-doubled to look unlike anything a
/// real workflow tag would emit. Falls back to a salted variant if the
/// input itself contains the prefix (see [`pick_placeholder_prefix`]).
const DEFAULT_PREFIX: &str = "{{HEADROOM_TAG_";
const PLACEHOLDER_SUFFIX: &str = "}}";

/// Sidecar diagnostics — same shape every Rust transform uses.
#[derive(Debug, Default, Clone)]
pub struct ProtectStats {
    pub tags_seen: usize,
    pub html_tags_skipped: usize,
    pub custom_blocks_protected: usize,
    pub self_closing_protected: usize,
    /// Closes that didn't match any open on the stack (malformed input
    /// or HTML interleaving). Emitted verbatim. Non-zero is a smell
    /// worth tracking but not necessarily a bug.
    pub orphan_closes: usize,
    /// True iff the placeholder prefix had to be salted because the
    /// input contained a literal `{{HEADROOM_TAG_` substring.
    pub placeholder_collision_avoided: bool,
}

/// Case-insensitive HTML tag check. Lowercases the input lazily so we
/// don't allocate for the common ASCII-lowercase case.
pub fn is_known_html_tag(tag_name: &str) -> bool {
    let set = known_html_tags();
    if set.contains(tag_name) {
        return true;
    }
    if tag_name.bytes().any(|b| b.is_ascii_uppercase()) {
        let lower = tag_name.to_ascii_lowercase();
        return set.contains(lower.as_str());
    }
    false
}

/// Iterate the canonical HTML tag list. Used by the PyO3 shim to
/// expose `KNOWN_HTML_TAGS` to Python without re-declaring the set.
pub fn known_html_tag_names() -> &'static [&'static str] {
    HTML5_TAGS
}

/// Pick a placeholder prefix that doesn't collide with anything in
/// `text`. We try `{{HEADROOM_TAG_` first; if the input contains it
/// literally we salt with a per-call counter until we miss. The salt
/// is bounded; in practice we never need more than one attempt.
fn pick_placeholder_prefix(text: &str) -> (String, bool) {
    if !text.contains(DEFAULT_PREFIX) {
        return (DEFAULT_PREFIX.to_string(), false);
    }
    for salt in 0u32..16 {
        let candidate = format!("{{{{HEADROOM_TAG_{salt}_");
        if !text.contains(&candidate) {
            return (candidate, true);
        }
    }
    // 16 salt attempts collided — fall back to a UUID-shaped marker.
    // The OnceLock cache is so two consecutive calls in the same
    // process don't pay the formatting cost.
    static FALLBACK: OnceLock<String> = OnceLock::new();
    let prefix = FALLBACK
        .get_or_init(|| "{{HEADROOM_TAG_FALLBACK_a4f1c7e2_".to_string())
        .clone();
    (prefix, true)
}

#[derive(Debug)]
struct OpenTag {
    /// Lowercase name for case-insensitive close-matching.
    name_lower: String,
    /// Byte offset of the `<` that opened this tag.
    open_start: usize,
}

/// Outcome of a single `<…>` parse attempt at a given offset.
enum TagParse {
    /// Opening tag (`<name attr=…>`). `name_end` is exclusive.
    Open {
        name_end: usize,
        tag_end: usize,
        is_self_closing: bool,
    },
    /// Closing tag (`</name>`).
    Close { name_end: usize, tag_end: usize },
    /// Not a tag (e.g. `<` followed by whitespace or digit).
    NotTag,
}

/// Parse a `<…>` starting at `start`. Returns the byte offset of the
/// closing `>` (exclusive end of the tag) and the kind. Conservatively
/// rejects malformed shapes — we'd rather emit a `<` verbatim than
/// over-protect on bad input.
fn parse_tag_at(bytes: &[u8], start: usize) -> TagParse {
    debug_assert!(bytes[start] == b'<');
    let mut i = start + 1;
    let n = bytes.len();
    if i >= n {
        return TagParse::NotTag;
    }

    let is_close = bytes[i] == b'/';
    if is_close {
        i += 1;
    }
    // After consuming a possible '/' we may be at end-of-input
    // (e.g. literal `</` with nothing after). Guard the bounds
    // before indexing into `bytes[i]` for the name-start check —
    // proptest discovered the OOB on input `</`.
    if i >= n {
        return TagParse::NotTag;
    }
    let name_start = i;
    if !is_name_start(bytes[i]) {
        return TagParse::NotTag;
    }
    i += 1;
    while i < n && is_name_cont(bytes[i]) {
        i += 1;
    }
    let name_end = i;
    if name_end == name_start {
        return TagParse::NotTag;
    }

    if is_close {
        // Allow optional whitespace, then `>`.
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n || bytes[i] != b'>' {
            return TagParse::NotTag;
        }
        return TagParse::Close {
            name_end,
            tag_end: i + 1,
        };
    }

    // Opening tag: skip attributes until `>` (handle `/>` for
    // self-closing). Quoted attribute values can contain `>`; a
    // single-pass attribute lexer handles the common cases.
    let mut self_closing = false;
    while i < n {
        match bytes[i] {
            b'>' => {
                return TagParse::Open {
                    name_end,
                    tag_end: i + 1,
                    is_self_closing: self_closing,
                };
            }
            b'/' => {
                self_closing = true;
                i += 1;
            }
            b'"' | b'\'' => {
                let quote = bytes[i];
                i += 1;
                while i < n && bytes[i] != quote {
                    i += 1;
                }
                if i >= n {
                    return TagParse::NotTag;
                }
                i += 1;
                self_closing = false;
            }
            _ => {
                if bytes[i].is_ascii_whitespace() {
                    self_closing = false;
                }
                i += 1;
            }
        }
    }

    TagParse::NotTag
}

#[inline]
fn is_name_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline]
fn is_name_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':')
}

/// A single span that was identified as worth replacing.
///
/// In block mode every matched custom-tag span (open..=close) becomes
/// one Span and is replaced by a single placeholder; self-closing
/// custom tags become a Span covering just the tag bytes.
///
/// In marker-only mode each opening custom tag and each closing custom
/// tag becomes its own Span (the body between them is left visible to
/// the compressor).
#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    kind: SpanKind,
}

#[derive(Debug, Clone, Copy)]
enum SpanKind {
    /// Whole `<custom>…</custom>` block (block mode).
    Block,
    /// Self-closing `<custom/>` (block mode).
    SelfClosing,
    /// Opening `<custom>` marker (marker-only mode).
    OpenMarker,
    /// Closing `</custom>` marker (marker-only mode).
    CloseMarker,
}

/// Protect custom workflow tags from text compression.
///
/// * `compress_tagged_content = false` (default) — replace each entire
///   `<custom>…</custom>` span (including nested children) with a
///   single placeholder. Self-closing custom tags become a single
///   placeholder. The body between the markers is *not* exposed to
///   compression.
/// * `compress_tagged_content = true` — replace only the tag markers
///   (open and close emitted as separate placeholders) so the
///   compressor can squash content while the tag boundaries survive.
///
/// Returns `(cleaned, blocks, stats)` where `blocks` is a list of
/// `(placeholder, original)` pairs for [`restore_tags`]. The blocks
/// are listed in left-to-right order of appearance in the input, which
/// keeps the restore step trivial.
pub fn protect_tags(
    text: &str,
    compress_tagged_content: bool,
) -> (String, Vec<(String, String)>, ProtectStats) {
    let mut stats = ProtectStats::default();
    if text.is_empty() || !text.contains('<') {
        return (text.to_string(), Vec::new(), stats);
    }

    let (prefix, salted) = pick_placeholder_prefix(text);
    stats.placeholder_collision_avoided = salted;

    // Phase 1: walk once, classify every tag, build a list of spans
    // worth replacing. No output emitted yet — this is purely
    // discovery so we can decide which byte ranges to swap.
    let spans = identify_spans(text, compress_tagged_content, &mut stats);

    // Phase 2: emit. Walk the input once more, splicing placeholders
    // for span bytes and copying everything else verbatim. Because
    // `spans` is sorted left-to-right and non-overlapping (block mode
    // collapses nested matches into the outermost span; marker mode
    // emits open/close markers that are byte-disjoint by construction)
    // this is a straightforward scan.
    match emit_output(text, &spans, &prefix) {
        Some((cleaned, blocks)) => (cleaned, blocks, stats),
        // Should be unreachable — `identify_spans` returns spans whose
        // bytes are slices of `text`. If we ever fail to splice them
        // back, fall back to emitting the original.
        None => (text.to_string(), Vec::new(), stats),
    }
}

fn identify_spans(
    text: &str,
    compress_tagged_content: bool,
    stats: &mut ProtectStats,
) -> Vec<Span> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut spans: Vec<Span> = Vec::new();
    let mut stack: Vec<OpenTag> = Vec::new();

    let mut i = 0;
    while i < n {
        let b = bytes[i];
        if b != b'<' {
            // Skip ahead to the next `<`. We don't care about non-tag
            // bytes for span identification; they'll be copied verbatim
            // in phase 2.
            i = memchr(b'<', &bytes[i..]).map(|j| i + j).unwrap_or(n);
            continue;
        }

        match parse_tag_at(bytes, i) {
            TagParse::NotTag => {
                i += 1;
            }
            TagParse::Open {
                name_end,
                tag_end,
                is_self_closing,
            } => {
                stats.tags_seen += 1;
                let name = &text[i + 1..name_end];
                if is_known_html_tag(name) {
                    stats.html_tags_skipped += 1;
                    i = tag_end;
                    continue;
                }
                if is_self_closing {
                    spans.push(Span {
                        start: i,
                        end: tag_end,
                        kind: SpanKind::SelfClosing,
                    });
                    stats.self_closing_protected += 1;
                    i = tag_end;
                    continue;
                }
                if compress_tagged_content {
                    // Marker-only mode: emit the open as its own span
                    // *and* push the name on the stack so the close
                    // gets matched and emitted as its own span.
                    spans.push(Span {
                        start: i,
                        end: tag_end,
                        kind: SpanKind::OpenMarker,
                    });
                }
                // Both modes push to the stack so close-matching works.
                stack.push(OpenTag {
                    name_lower: name.to_ascii_lowercase(),
                    open_start: i,
                });
                i = tag_end;
            }
            TagParse::Close { name_end, tag_end } => {
                stats.tags_seen += 1;
                let close_name = &text[i + 2..name_end];
                if is_known_html_tag(close_name) {
                    stats.html_tags_skipped += 1;
                    i = tag_end;
                    continue;
                }
                let close_name_lower = close_name.to_ascii_lowercase();
                let matching = stack
                    .iter()
                    .rposition(|open| open.name_lower == close_name_lower);

                match matching {
                    Some(stack_idx) => {
                        if compress_tagged_content {
                            // Pop everything above (orphan opens
                            // inside the matched span — their open
                            // markers were already recorded as spans
                            // and we keep them).
                            stack.truncate(stack_idx);
                            let _ = stack.pop();
                            spans.push(Span {
                                start: i,
                                end: tag_end,
                                kind: SpanKind::CloseMarker,
                            });
                        } else {
                            // Block mode: collapse [open..close] into
                            // a single span. Drop any inner unmatched
                            // opens (they're part of this span's body).
                            // Also DROP any inner spans we already
                            // recorded that are now subsumed by this
                            // outer block — that's how nested custom
                            // tags collapse to one placeholder.
                            let open_start = stack[stack_idx].open_start;
                            stack.truncate(stack_idx);
                            spans.retain(|s| s.start < open_start);
                            spans.push(Span {
                                start: open_start,
                                end: tag_end,
                                kind: SpanKind::Block,
                            });
                            stats.custom_blocks_protected += 1;
                        }
                        i = tag_end;
                    }
                    None => {
                        stats.orphan_closes += 1;
                        i = tag_end;
                    }
                }
            }
        }
    }

    // Stack remnants are orphan opens (no matching close ever arrived).
    // We don't protect those — they'll fall through to the compressor
    // as raw `<name>` bytes, same as Python's original behavior. In
    // block mode their inner self-closing spans we recorded are still
    // safe to keep: they were below an unmatched outer open, so they
    // were never collapsed. Spans are sorted by start ascending due to
    // the monotonic walk; phase 2 expects that.
    spans
}

fn emit_output(
    text: &str,
    spans: &[Span],
    prefix: &str,
) -> Option<(String, Vec<(String, String)>)> {
    let mut out = String::with_capacity(text.len());
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut cursor: usize = 0;

    for (counter, span) in (0_u64..).zip(spans.iter()) {
        if span.start < cursor {
            // Overlap shouldn't happen given how we collapse nested
            // spans, but bail loudly if it does — silently producing
            // wrong output is worse than failing the test.
            return None;
        }
        out.push_str(&text[cursor..span.start]);
        let placeholder = format!("{prefix}{counter}{PLACEHOLDER_SUFFIX}");
        let original = &text[span.start..span.end];
        blocks.push((placeholder.clone(), original.to_string()));
        out.push_str(&placeholder);
        cursor = span.end;
        let _ = span.kind; // SpanKind is informational only at this layer
    }
    out.push_str(&text[cursor..]);
    Some((out, blocks))
}

/// Restore protected tag spans after the compressor ran on the
/// cleaned text.
///
/// # Hotfix-A9 — discard-wrap semantics
///
/// If a placeholder went missing during compression (the compressor
/// stripped or rewrote it) the wrap is **discarded**: the compressed
/// text flows downstream as-is and the original tag bytes are NOT
/// re-injected anywhere. This is a deliberate behavior change vs the
/// original "append the orphan tag at the trailing edge" fallback,
/// which produced silently malformed XML (an opening tag with no
/// closing tag and no body) on ~350 production requests over 9 days.
///
/// An ERROR-level `tracing` event with structured fields
/// (`event = tag_protector_placeholder_lost`, `tag_preview`,
/// `compressed_length`, optional `request_id`) is emitted per lost
/// placeholder so operators can alert on the corruption rather than
/// having it disappear into a WARN line. Token validation downstream
/// is responsible for catching cases where the discard regressed the
/// final output vs the original input.
///
/// Invariants enforced:
/// 1. Symmetry — never emit asymmetric tag counts (the input either
///    survives both opens and closes via successful substitution, or
///    neither survives via discard).
/// 2. No orphan tag injection — `restore_tags` adds bytes only as part
///    of placeholder substitution. No appends, no prepends, no
///    whitespace insertion outside placeholder substitutions.
/// 3. Idempotence on missing placeholders — if every placeholder is
///    absent from `compressed`, the function returns `compressed`
///    byte-for-byte unchanged.
pub fn restore_tags(text: &str, blocks: &[(String, String)]) -> String {
    restore_tags_with_request_id(text, blocks, None)
}

/// Variant of [`restore_tags`] that threads an optional `request_id`
/// into the structured ERROR log emitted on placeholder loss. The
/// PyO3 binding currently calls [`restore_tags`] (no request id);
/// this entry point exists so the proxy layer can wire request
/// context through once it has one available end-to-end.
pub fn restore_tags_with_request_id(
    text: &str,
    blocks: &[(String, String)],
    request_id: Option<&str>,
) -> String {
    if blocks.is_empty() {
        return text.to_string();
    }

    let mut result = text.to_string();
    let mut lost_count: usize = 0;
    let compressed_length = text.len();
    for (placeholder, original) in blocks {
        if result.contains(placeholder.as_str()) {
            result = result.replace(placeholder.as_str(), original);
        } else {
            lost_count += 1;
            tag_lost_error(original, compressed_length, request_id);
        }
    }
    // No tail_appends. The compressed text is returned with the
    // wraps for the lost placeholders fully discarded — never
    // appended back as orphan opens (Hotfix-A9).
    let _ = lost_count; // surfaced via the per-event log; aggregate
                        // counters live on the stats sidecar in
                        // callers that have one.
    result
}

#[inline(never)]
fn tag_lost_error(original: &str, compressed_length: usize, request_id: Option<&str>) {
    let preview: String = original.chars().take(80).collect();
    match request_id {
        Some(rid) => tracing::error!(
            target: "headroom::tag_protector",
            event = "tag_protector_placeholder_lost",
            tag_preview = %preview,
            compressed_length = compressed_length,
            request_id = %rid,
            action = "discarded_wrap",
            "tag placeholder lost during compression — wrap discarded"
        ),
        None => tracing::error!(
            target: "headroom::tag_protector",
            event = "tag_protector_placeholder_lost",
            tag_preview = %preview,
            compressed_length = compressed_length,
            action = "discarded_wrap",
            "tag placeholder lost during compression — wrap discarded"
        ),
    }
}

// ─── Tiny byte-search helper ──────────────────────────────────────────

#[inline]
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protect(text: &str) -> (String, Vec<(String, String)>) {
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        (cleaned, blocks)
    }

    #[test]
    fn passthrough_when_no_angle_bracket() {
        let (cleaned, blocks) = protect("Just plain text");
        assert_eq!(cleaned, "Just plain text");
        assert!(blocks.is_empty());
    }

    #[test]
    fn html_tags_emitted_verbatim() {
        let text = "<div>Some content</div>";
        let (cleaned, blocks) = protect(text);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn html_tag_check_case_insensitive() {
        assert!(is_known_html_tag("DIV"));
        assert!(is_known_html_tag("Span"));
        assert!(!is_known_html_tag("system-reminder"));
        assert!(!is_known_html_tag("EXTREMELY_IMPORTANT"));
    }

    #[test]
    fn custom_tag_replaced_with_placeholder() {
        let text = "Before <system-reminder>Important</system-reminder> After";
        let (cleaned, blocks) = protect(text);
        assert!(!cleaned.contains("<system-reminder>"));
        assert!(!cleaned.contains("Important"));
        assert!(cleaned.contains("Before"));
        assert!(cleaned.contains("After"));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<system-reminder>Important</system-reminder>");
    }

    #[test]
    fn custom_tag_with_attributes() {
        let text = r#"<context key="session" type="persistent">user data</context>"#;
        let (_cleaned, blocks) = protect(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].1.contains(r#"key="session""#));
    }

    #[test]
    fn self_closing_custom_tag() {
        let text = "Text <marker/> more text";
        let (_cleaned, blocks) = protect(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<marker/>");
    }

    #[test]
    fn self_closing_html_tag_not_protected() {
        let text = "Text <br/> more <hr/> text";
        let (cleaned, blocks) = protect(text);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn nested_custom_tags_collapse_to_outer_span() {
        let text = "<outer><inner>deep</inner></outer>";
        let (cleaned, blocks) = protect(text);
        assert!(!cleaned.contains("<outer>"));
        assert!(!cleaned.contains("<inner>"));
        // Outer span captures inner — single placeholder.
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<outer><inner>deep</inner></outer>");
    }

    #[test]
    fn mixed_html_and_custom() {
        let text = "<div>HTML</div> <system-reminder>Rule</system-reminder> <p>HTML2</p>";
        let (cleaned, blocks) = protect(text);
        assert!(cleaned.contains("<div>"));
        assert!(cleaned.contains("<p>"));
        assert!(!cleaned.contains("<system-reminder>"));
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn real_workflow_tags() {
        let cases = [
            "<tool_call>search({query: 'test'})</tool_call>",
            "<thinking>Let me analyze this</thinking>",
            "<EXTREMELY_IMPORTANT>Never skip validation</EXTREMELY_IMPORTANT>",
            "<user-prompt-submit-hook>check perms</user-prompt-submit-hook>",
            "<system-reminder>Rules</system-reminder>",
            "<result>Success: 42 items</result>",
        ];
        for tag in cases {
            let text = format!("Before {tag} After");
            let (_cleaned, blocks) = protect(&text);
            assert_eq!(blocks.len(), 1, "failed to protect: {tag}");
            assert_eq!(blocks[0].1, tag);
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let (cleaned, blocks) = protect("");
        assert!(cleaned.is_empty());
        assert!(blocks.is_empty());
    }

    #[test]
    fn compress_tagged_content_true_emits_marker_placeholders() {
        let text = "Before <system-reminder>Compressible content</system-reminder> After";
        let (cleaned, blocks, _stats) = protect_tags(text, true);
        assert!(!cleaned.contains("<system-reminder>"));
        assert!(!cleaned.contains("</system-reminder>"));
        assert!(cleaned.contains("Compressible content"));
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn restore_basic() {
        let original = "Before <system-reminder>Rule</system-reminder> After";
        let (cleaned, blocks, _stats) = protect_tags(original, false);
        let restored = restore_tags(&cleaned, &blocks);
        assert_eq!(restored, original);
    }

    #[test]
    fn restore_empty_blocks_passthrough() {
        assert_eq!(restore_tags("untouched", &[]), "untouched");
    }

    #[test]
    fn restore_lost_placeholder_discards_wrap() {
        // Hotfix-A9: when a placeholder is missing from the compressed
        // text, the wrap is DISCARDED — the compressed text is returned
        // as-is, with no orphan-tag append. (The original behavior of
        // appending the tag at the trailing edge produced silently
        // malformed XML in ~350 production requests over 9 days.)
        let blocks = vec![(
            "{{HEADROOM_TAG_0}}".to_string(),
            "<tag>data</tag>".to_string(),
        )];
        let compressed = "text without placeholder";
        let restored = restore_tags(compressed, &blocks);
        // Compressed text returned unchanged; original tag NOT injected.
        assert_eq!(restored, compressed);
        assert!(!restored.contains("<tag>"));
        assert!(!restored.contains("</tag>"));
        assert!(!restored.contains("<tag>data</tag>"));
    }

    #[test]
    fn restore_lost_placeholder_idempotent_when_all_missing() {
        // Invariant #3: if every placeholder is missing from the
        // compressed text, the function returns the compressed text
        // byte-for-byte unchanged.
        let blocks = vec![
            ("{{HEADROOM_TAG_0}}".to_string(), "<a>1</a>".to_string()),
            ("{{HEADROOM_TAG_1}}".to_string(), "<b>2</b>".to_string()),
            ("{{HEADROOM_TAG_2}}".to_string(), "<c>3</c>".to_string()),
        ];
        let compressed = "compressor stripped every placeholder";
        let restored = restore_tags(compressed, &blocks);
        assert_eq!(restored, compressed);
    }

    #[test]
    fn restore_partial_loss_keeps_present_drops_lost() {
        // Mixed case: some placeholders survive, others are lost. The
        // surviving ones get substituted; the lost ones are discarded.
        // No orphan-tag bytes appear anywhere in the output.
        let blocks = vec![
            ("{{HEADROOM_TAG_0}}".to_string(), "<a>1</a>".to_string()),
            (
                "{{HEADROOM_TAG_1}}".to_string(),
                "<lost>x</lost>".to_string(),
            ),
        ];
        let compressed = "head {{HEADROOM_TAG_0}} tail";
        let restored = restore_tags(compressed, &blocks);
        assert_eq!(restored, "head <a>1</a> tail");
        assert!(!restored.contains("<lost"));
        assert!(!restored.contains("</lost>"));
    }

    #[test]
    fn restore_roundtrip_preserves_content() {
        let original = "Start <system-reminder>Rule 1: validate</system-reminder> middle \
             <tool_call>search(q='test')</tool_call> end";
        let (cleaned, blocks, _stats) = protect_tags(original, false);
        let restored = restore_tags(&cleaned, &blocks);
        assert_eq!(restored, original);
    }

    // ─── Bug-fix tests (fixed_in_3e4) ─────────────────────────────────

    #[test]
    fn fixed_in_3e4_replace_first_does_not_collide_on_duplicate_blocks() {
        // Bug #2: Python's `result.replace(original, placeholder, 1)`
        // replaces the FIRST textual occurrence of `original`, not
        // necessarily the matched offset. Two identical custom-tag
        // blocks would collapse to a single placeholder + a stray
        // duplicate of the second block in the output.
        let text = "<system-reminder>same</system-reminder> middle \
             <system-reminder>same</system-reminder>";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        // BOTH blocks should be replaced by DIFFERENT placeholders.
        assert_eq!(blocks.len(), 2);
        assert!(!cleaned.contains("<system-reminder>"));
        assert!(!cleaned.contains("</system-reminder>"));
        assert_ne!(blocks[0].0, blocks[1].0);
        // Roundtrip is exact.
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn fixed_in_3e4_handles_50_plus_nested_custom_tags() {
        // Bug #3: Python had a hard-coded 50-iteration safety cap that
        // silently truncated tag protection on deeply nested input.
        // Build 60 nested custom tags and verify all get caught in
        // the outermost span.
        let depth = 60;
        let mut text = String::new();
        for _ in 0..depth {
            text.push_str("<lvl>");
        }
        text.push_str("core");
        for _ in 0..depth {
            text.push_str("</lvl>");
        }
        let (cleaned, blocks, _stats) = protect_tags(&text, false);
        // The outermost span eats everything: one placeholder, no
        // residual `<lvl>` markers in the cleaned text.
        assert!(!cleaned.contains("<lvl>"));
        assert!(!cleaned.contains("</lvl>"));
        assert_eq!(blocks.len(), 1);
        // Roundtrip exact even at depth=60.
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn fixed_in_3e4_self_closing_duplicates_get_distinct_placeholders() {
        // Bug #4: same first-occurrence-replace bug for self-closing
        // tags. `<marker/>` appearing twice would collapse.
        let text = "<marker/> middle <marker/>";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert_eq!(blocks.len(), 2);
        assert_ne!(blocks[0].0, blocks[1].0);
        assert!(!cleaned.contains("<marker/>"));
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn fixed_in_3e4_placeholder_collision_is_avoided() {
        // Bug #5: input contains literal `{{HEADROOM_TAG_…}}`. The
        // walker should pick a salted prefix and report the collision
        // in stats.
        let text = "User wrote {{HEADROOM_TAG_0}} on purpose. \
             <system-reminder>real one</system-reminder>";
        let (_cleaned, blocks, stats) = protect_tags(text, false);
        assert!(stats.placeholder_collision_avoided);
        assert_eq!(blocks.len(), 1);
        // Placeholder used must NOT collide with the user's literal.
        assert_ne!(blocks[0].0, "{{HEADROOM_TAG_0}}");
    }

    // ─── Edge-case correctness ────────────────────────────────────────

    #[test]
    fn orphan_close_tag_emitted_verbatim() {
        let text = "no opener </ghost> here";
        let (cleaned, blocks, stats) = protect_tags(text, false);
        // Nothing protected; close stays in the cleaned text.
        assert_eq!(blocks.len(), 0);
        assert!(cleaned.contains("</ghost>"));
        assert_eq!(stats.orphan_closes, 1);
    }

    #[test]
    fn orphan_open_tag_emitted_verbatim() {
        // An open with no matching close should round-trip exactly —
        // no protection, no data loss.
        let text = "<ghost>dangling content with no close";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert!(blocks.is_empty());
        assert_eq!(cleaned, text);
    }

    #[test]
    fn malformed_lone_lt_emitted_verbatim() {
        let text = "if a < b then c";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn truncated_close_marker_does_not_panic() {
        // Hotfix-A9: proptest seed `</` would index past end-of-input
        // in `parse_tag_at`. Pre-fix this panicked with an OOB; the
        // bounds-check now returns NotTag and the function falls
        // through to emitting `</` verbatim.
        for text in ["</", "<", "<a/", "<a", "<a /", "</a"] {
            let (cleaned, blocks, _stats) = protect_tags(text, false);
            assert_eq!(cleaned, text);
            assert!(blocks.is_empty());
        }
    }

    #[test]
    fn attribute_with_gt_inside_quotes() {
        let text = r#"<context attr="a > b">payload</context>"#;
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, text);
        assert!(!cleaned.contains("payload"));
    }

    #[test]
    fn html_close_inside_custom_block_does_not_pop_stack() {
        // An HTML close tag while a custom open is on top should not
        // confuse the stack: the HTML close is emitted verbatim, the
        // custom span still closes when its own close arrives.
        let text = "<custom>x</div> y</custom>";
        let (cleaned, blocks, stats) = protect_tags(text, false);
        // The whole `<custom>...</custom>` span wins, including the
        // verbatim `</div>` inside.
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<custom>x</div> y</custom>");
        assert!(!cleaned.contains("<custom>"));
        // `</div>` is HTML, not orphan.
        assert_eq!(stats.html_tags_skipped, 1);
        assert_eq!(stats.orphan_closes, 0);
    }

    // ─── Hotfix-A9 invariants ────────────────────────────────────────

    /// Count `<custom>` style opening tags (excludes self-closers and
    /// excludes the closing-tag `</…>` form). Any `<` that is followed
    /// by an alphabetic name and ends with `>` (without an embedded
    /// `/>`) counts. Only used by the proptest below — keeps the
    /// invariant check independent of the parser under test.
    fn count_open_tags(s: &str) -> usize {
        let bytes = s.as_bytes();
        let mut count = 0_usize;
        let mut i = 0_usize;
        while i < bytes.len() {
            if bytes[i] != b'<' {
                i += 1;
                continue;
            }
            // Skip closing tags `</…>`.
            if i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                i += 1;
                continue;
            }
            // Must be followed by a name-start char to count as a tag.
            if i + 1 >= bytes.len() || !is_name_start(bytes[i + 1]) {
                i += 1;
                continue;
            }
            // Walk to the matching `>`. If we hit `/>` first, this is
            // self-closing and doesn't count as an unbalanced opener.
            let mut j = i + 1;
            let mut self_closing = false;
            while j < bytes.len() && bytes[j] != b'>' {
                if bytes[j] == b'/' {
                    self_closing = true;
                }
                j += 1;
            }
            if j >= bytes.len() {
                // No closing `>` — not a tag.
                break;
            }
            if !self_closing {
                count += 1;
            }
            i = j + 1;
        }
        count
    }

    fn count_close_tags(s: &str) -> usize {
        let bytes = s.as_bytes();
        let mut count = 0_usize;
        let mut i = 0_usize;
        while i < bytes.len() {
            if bytes[i] != b'<' {
                i += 1;
                continue;
            }
            if i + 1 >= bytes.len() || bytes[i + 1] != b'/' {
                i += 1;
                continue;
            }
            // `</name>` — confirm name-start and find the closing `>`.
            if i + 2 >= bytes.len() || !is_name_start(bytes[i + 2]) {
                i += 1;
                continue;
            }
            let mut j = i + 2;
            while j < bytes.len() && bytes[j] != b'>' {
                j += 1;
            }
            if j >= bytes.len() {
                break;
            }
            count += 1;
            i = j + 1;
        }
        count
    }

    proptest::proptest! {
        /// Invariant: `restore_tags` never INTRODUCES tag-count
        /// asymmetry. Concretely: restoring on a compressed text with
        /// any subset of placeholders missing must produce the same
        /// `opens - closes` skew as the cleaned text after stripping
        /// the placeholders. The orphan-append bug fixed by Hotfix-A9
        /// could turn a symmetric `<a>x</a>` into an asymmetric
        /// `compressed-stuff <a>` whenever the placeholder was
        /// dropped — the discard-wrap path makes that impossible
        /// because every protected span is a balanced wrap (or a
        /// self-closer) so dropping it changes opens and closes by
        /// the same amount.
        #[test]
        fn restore_never_introduces_asymmetry(content in "[a-z<>/]{0,200}") {
            let (cleaned, blocks, _stats) = protect_tags(&content, false);
            // Baseline: strip every placeholder from `cleaned`. This
            // is the "lost everything" worst case; the discard-wrap
            // path must produce exactly this output.
            let mut stripped = cleaned.clone();
            for (placeholder, _original) in &blocks {
                stripped = stripped.replace(placeholder.as_str(), "");
            }
            let baseline_skew = count_open_tags(&stripped) as i64
                - count_close_tags(&stripped) as i64;

            // With every placeholder lost, restore_tags must return
            // the compressed text with placeholders dropped — which
            // is exactly `stripped`. So asymmetry equals baseline.
            let restored_all_lost = restore_tags(&stripped, &blocks);
            let lost_skew = count_open_tags(&restored_all_lost) as i64
                - count_close_tags(&restored_all_lost) as i64;
            proptest::prop_assert_eq!(
                lost_skew, baseline_skew,
                "discard-wrap path introduced asymmetry: baseline={}, after_restore={}, restored={:?}",
                baseline_skew, lost_skew, restored_all_lost
            );

            // With every placeholder PRESENT, restore_tags must round-
            // trip exactly to the original `content`, which by
            // construction has the same skew as `content` itself.
            let restored_full = restore_tags(&cleaned, &blocks);
            let full_skew = count_open_tags(&restored_full) as i64
                - count_close_tags(&restored_full) as i64;
            let content_skew = count_open_tags(&content) as i64
                - count_close_tags(&content) as i64;
            proptest::prop_assert_eq!(
                full_skew, content_skew,
                "full-restore path drifted from input skew: input={}, restored={}",
                content_skew, full_skew
            );
        }

        /// Invariant: when every placeholder is stripped before
        /// restore, the function returns the compressed text
        /// byte-for-byte unchanged (no orphan-tag injection, no
        /// whitespace insertion, no prepends/appends).
        #[test]
        fn restore_idempotent_when_all_placeholders_lost(
            content in "[a-z<>/]{0,200}",
            compressed in "[ -~]{0,200}",
        ) {
            let (_cleaned, blocks, _stats) = protect_tags(&content, false);
            // Drop all placeholders by feeding `restore_tags` arbitrary
            // text the compressor "produced". If none of the
            // placeholders happen to appear in `compressed` (the
            // common case for arbitrary strings), the discard-wrap
            // path runs end-to-end.
            let any_placeholder_present = blocks
                .iter()
                .any(|(p, _)| compressed.contains(p.as_str()));
            proptest::prop_assume!(!any_placeholder_present);
            let restored = restore_tags(&compressed, &blocks);
            proptest::prop_assert_eq!(restored, compressed);
        }

        /// Invariant: `restore_tags` never adds bytes that weren't
        /// already in `compressed` or part of a substituted placeholder
        /// original. Concretely: the restored length is at most
        /// `compressed.len()` plus the sum of lengths of originals
        /// that actually got substituted; lost-placeholder originals
        /// contribute zero bytes.
        #[test]
        fn restore_no_orphan_byte_injection(
            content in "[a-z<>/]{0,200}",
        ) {
            let (cleaned, blocks, _stats) = protect_tags(&content, false);
            let restored = restore_tags(&cleaned, &blocks);
            // Sum of the byte-lengths of the originals that were
            // actually substituted (placeholder still present in
            // `cleaned`). Lost placeholders contribute zero.
            let substituted_original_bytes: usize = blocks
                .iter()
                .filter(|(p, _)| cleaned.contains(p.as_str()))
                .map(|(p, original)| original.len().saturating_sub(p.len()))
                .sum();
            // Upper bound: cleaned.len() + delta from substitution.
            // (Substitution replaces each placeholder of len p.len()
            // with original of len original.len(); delta per substitution
            // is original.len() - p.len(), summed across substitutions.)
            let upper_bound = cleaned.len() + substituted_original_bytes;
            proptest::prop_assert!(
                restored.len() <= upper_bound,
                "restored too long: restored.len={} upper_bound={} cleaned.len={}",
                restored.len(), upper_bound, cleaned.len()
            );
        }
    }
}
