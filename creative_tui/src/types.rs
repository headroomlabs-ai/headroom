//! Data layer — all types with no IO, no framework dependencies.
//!
//! Every type that crosses a layer boundary lives here. No raw strings
//! in channels, no unconstrained floats. Newtypes carry their
//! validation in the constructor.

use std::collections::VecDeque;
use std::fmt;

// ─── domain newtypes ──────────────────────────────────────────────

/// Animation speed multiplier. Guaranteed in [0.1, 10.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Speed(f32);

impl Speed {
    pub const MIN: f32 = 0.1;
    pub const MAX: f32 = 10.0;
    pub const DEFAULT: Self = Self(1.0);

    pub fn new(v: f32) -> Self {
        Self(v.clamp(Self::MIN, Self::MAX))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Speed {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for Speed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}", self.0)
    }
}

impl From<f32> for Speed {
    fn from(v: f32) -> Self {
        Self::new(v)
    }
}

/// RGB colour multiplier for shader tint. Each channel in [0.0, 3.0].
/// Values above 1.0 amplify the channel; below 1.0 dim it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    pub const MIN: f32 = 0.0;
    pub const MAX: f32 = 3.0;

    pub const DEFAULT: Self = Self {
        r: 1.0,
        g: 0.8,
        b: 0.6,
    };

    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(Self::MIN, Self::MAX),
            g: g.clamp(Self::MIN, Self::MAX),
            b: b.clamp(Self::MIN, Self::MAX),
        }
    }
}

impl Default for Rgb {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.r, self.g, self.b)
    }
}

// ─── shader uniforms (GPU-facing, repr(C) for bytemuck) ──────────

/// Shader uniform block — passed to the GPU every frame.
/// Must be `#[repr(C)]` for bytemuck zero-copy GPU upload.
///
/// Raw `f32` fields are intentional — GPU shaders consume floats.
/// Use [`Speed`] and [`Rgb`] for validated construction; call
/// [`Uniforms::apply`] to transfer them.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub time: f32,
    pub speed: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
}

impl Uniforms {
    /// Apply validated domain types to the raw GPU fields.
    pub fn apply_speed(&mut self, speed: Speed) {
        self.speed = speed.get();
    }

    /// Apply validated colour to the raw GPU fields.
    pub fn apply_color(&mut self, rgb: Rgb) {
        self.color_r = rgb.r;
        self.color_g = rgb.g;
        self.color_b = rgb.b;
    }

    /// Snapshot current GPU values back into domain types.
    pub fn speed_val(&self) -> Speed {
        Speed::new(self.speed)
    }

    /// Snapshot current GPU values back into domain types.
    pub fn color_val(&self) -> Rgb {
        Rgb::new(self.color_r, self.color_g, self.color_b)
    }

    /// Set elapsed time (called every frame by renderer).
    pub fn set_time(&mut self, t: f32) {
        self.time = t;
    }
}

impl Default for Uniforms {
    fn default() -> Self {
        let speed = Speed::default();
        let rgb = Rgb::default();
        Self {
            time: 0.0,
            speed: speed.get(),
            color_r: rgb.r,
            color_g: rgb.g,
            color_b: rgb.b,
        }
    }
}

// ─── commands (TUI → shell dispatcher) ──────────────────────────

/// A parsed command from the TUI shell prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    SetSpeed(Speed),
    SetColor(Rgb),
    ShowTime,
}

impl Command {
    /// Parse a raw input line. Returns `None` for unrecognised input.
    pub fn parse(input: &str) -> Option<Self> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.as_slice() {
            ["speed", v] => v.parse::<f32>().ok().map(|n| Command::SetSpeed(Speed::new(n))),
            ["color", r, g, b] => {
                let rv = r.parse::<f32>().ok()?;
                let gv = g.parse::<f32>().ok()?;
                let bv = b.parse::<f32>().ok()?;
                Some(Command::SetColor(Rgb::new(rv, gv, bv)))
            }
            ["time"] => Some(Command::ShowTime),
            _ => None,
        }
    }
}

// ─── shell responses (shell dispatcher → TUI log) ───────────────

/// A typed response from the command handler.
#[derive(Debug, Clone)]
pub enum ShellResponse {
    Ok(String),
    Info(String),
    Err(String),
}

impl ShellResponse {
    pub fn ok(msg: impl fmt::Display) -> Self {
        Self::Ok(msg.to_string())
    }

    pub fn info(msg: impl fmt::Display) -> Self {
        Self::Info(msg.to_string())
    }

    pub fn err(msg: impl fmt::Display) -> Self {
        Self::Err(msg.to_string())
    }
}

impl fmt::Display for ShellResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShellResponse::Ok(msg) => write!(f, "[OK] {}", msg),
            ShellResponse::Info(msg) => write!(f, "[INFO] {}", msg),
            ShellResponse::Err(msg) => write!(f, "[ERR] {}", msg),
        }
    }
}

// ─── log buffer (ring buffer, no-clone view) ─────────────────────

/// Fixed-capacity ring buffer for TUI log lines.
/// Stores up to `capacity` entries; older entries are silently dropped.
/// `view()` returns an iterator without cloning.
pub struct LogBuffer {
    inner: VecDeque<String>,
    capacity: usize,
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a line; evict oldest if at capacity.
    pub fn push(&mut self, line: impl fmt::Display) {
        if self.inner.len() >= self.capacity {
            self.inner.pop_front();
        }
        self.inner.push_back(line.to_string());
    }

    /// Non-allocating view of the last N entries (capped at capacity).
    /// Returns lines newest-first for TUI rendering.
    pub fn view_recent(&self, n: usize) -> impl Iterator<Item = &str> + '_ {
        let skip = self.inner.len().saturating_sub(n);
        self.inner.iter().skip(skip).map(String::as_str)
    }

    /// Number of entries currently stored.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

// ─── window dimensions ───────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct WindowSize {
    pub width: u32,
    pub height: u32,
}

impl WindowSize {
    pub const DEFAULT: Self = Self {
        width: 800,
        height: 600,
    };
}

// ─── configuration ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub window_size: WindowSize,
    pub window_title: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            window_size: WindowSize::DEFAULT,
            window_title: "creative-tui -- shader canvas".into(),
        }
    }
}

