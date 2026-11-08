//! `headroom-xray` finds npx and hands CodeBurn its arguments and exit code.
//!
//! A fake npx stands in for npm's launcher: `npx` (a shell script) on Unix and
//! `npx.cmd` on Windows, where a bare `npx` is never found. It answers
//! `--version`, records the arguments it was given, and exits 7.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use headroom_xray::codeburn::PINNED_CODEBURN_VERSION;

#[cfg(unix)]
fn write_fakes(dir: &Path, record: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let scripts = [
        ("node", "#!/bin/sh\necho v20.11.0\n".to_string()),
        (
            "npx",
            format!(
                "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 10.9.4; exit 0; fi\n\
                 for a in \"$@\"; do echo \"$a\" >> '{}'; done\nexit 7\n",
                record.display()
            ),
        ),
    ];
    for (name, body) in scripts {
        let path = dir.join(name);
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
}

#[cfg(windows)]
fn write_fakes(dir: &Path, record: &Path) {
    // Real node comes from the runner's PATH; only npx is faked.
    let body = format!(
        "@echo off\r\nif \"%~1\"==\"--version\" (echo 10.9.4& exit /b 0)\r\n\
         :next\r\nif \"%~1\"==\"\" exit /b 7\r\n\
         >>\"{}\" echo %~1\r\nshift\r\ngoto next\r\n",
        record.display()
    );
    fs::write(dir.join("npx.cmd"), body).unwrap();
}

#[test]
fn delegates_to_npx_with_argv_and_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let record = dir.path().join("argv.txt");
    write_fakes(dir.path(), &record);
    let path = if cfg!(windows) {
        let inherited = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![dir.path().to_path_buf()];
        paths.extend(std::env::split_paths(&inherited));
        std::env::join_paths(paths).unwrap()
    } else {
        dir.path().as_os_str().to_owned()
    };

    Command::cargo_bin("headroom-xray")
        .unwrap()
        .env("PATH", path)
        .args(["report", "--period", "7d"])
        .assert()
        .code(7);

    let argv: Vec<String> = fs::read_to_string(&record)
        .unwrap()
        .lines()
        .map(|line| line.trim().to_string())
        .collect();
    assert_eq!(
        argv,
        [
            format!("codeburn@{PINNED_CODEBURN_VERSION}"),
            "report".into(),
            "--period".into(),
            "7d".into()
        ]
    );
}
