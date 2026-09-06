//! Shell — pure command dispatcher. No IO, no framework.
//!
//! Takes a [`Command`], mutates [`Uniforms`], returns a [`ShellResponse`].

use crate::types::{Command, ShellResponse, Uniforms};

/// Apply a command to the shared uniform state, returning a response
/// for the TUI log. This is a pure function — the caller owns locking.
pub fn dispatch(command: Command, uniforms: &mut Uniforms) -> ShellResponse {
    match command {
        Command::SetSpeed(speed) => {
            uniforms.apply_speed(speed);
            ShellResponse::ok(format_args!("speed = {}", speed))
        }
        Command::SetColor(rgb) => {
            uniforms.apply_color(rgb);
            ShellResponse::ok(format_args!("color = {}", rgb))
        }
        Command::ShowTime => ShellResponse::info(format_args!("time = {:.2}", uniforms.time)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Rgb, Speed};

    #[test]
    fn parse_speed() {
        assert_eq!(
            Command::parse("speed 3.5"),
            Some(Command::SetSpeed(Speed::new(3.5)))
        );
        assert_eq!(Command::parse("speed abc"), None);
        assert_eq!(Command::parse("speed"), None);
        // clamp
        assert_eq!(
            Command::parse("speed 999"),
            Some(Command::SetSpeed(Speed::new(10.0)))
        );
    }

    #[test]
    fn parse_color() {
        assert_eq!(
            Command::parse("color 1.0 0.5 0.2"),
            Some(Command::SetColor(Rgb::new(1.0, 0.5, 0.2)))
        );
        assert_eq!(Command::parse("color 1.0 0.5"), None);
    }

    #[test]
    fn parse_time() {
        assert_eq!(Command::parse("time"), Some(Command::ShowTime));
    }

    #[test]
    fn parse_unknown() {
        assert_eq!(Command::parse("foobar"), None);
        assert_eq!(Command::parse(""), None);
    }

    #[test]
    fn dispatch_speed_clamps_via_newtype() {
        let mut u = Uniforms::default();
        dispatch(Command::SetSpeed(Speed::new(-5.0)), &mut u);
        assert_eq!(u.speed, 0.1);
        dispatch(Command::SetSpeed(Speed::new(50.0)), &mut u);
        assert_eq!(u.speed, 10.0);
    }

    #[test]
    fn dispatch_color_roundtrip() {
        let mut u = Uniforms::default();
        let r = dispatch(Command::SetColor(Rgb::new(0.5, 0.6, 0.7)), &mut u);
        assert_eq!(u.color_r, 0.5);
        assert_eq!(u.color_g, 0.6);
        assert_eq!(u.color_b, 0.7);
        assert!(r.to_string().contains("0.50, 0.60, 0.70"));
    }
}
