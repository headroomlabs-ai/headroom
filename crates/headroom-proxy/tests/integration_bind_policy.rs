//! Startup bind policy, driven through the real binary.
//!
//! Same rule as the Python proxy (`headroom/proxy/bind_policy.py`): a
//! non-loopback `--listen` with no `HEADROOM_PROXY_TOKEN` must not start
//! unless `HEADROOM_ALLOW_UNAUTHENTICATED_BIND` acknowledges it.

use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

fn proxy(listen: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_headroom-proxy"));
    cmd.args(["--listen", listen, "--upstream", "http://127.0.0.1:9"])
        .env_remove("HEADROOM_PROXY_TOKEN")
        .env_remove("HEADROOM_ALLOW_UNAUTHENTICATED_BIND")
        .env_remove("HEADROOM_PROXY_LISTEN")
        .env("HEADROOM_PROXY_STATS", "false")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd
}

#[tokio::test]
async fn non_loopback_listen_without_token_refuses_to_start() {
    let child = proxy("0.0.0.0:0").spawn().expect("spawn proxy");
    let out = tokio::time::timeout(Duration::from_secs(20), child.wait_with_output())
        .await
        .expect("proxy kept running on 0.0.0.0 with no inbound token")
        .expect("wait");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("HEADROOM_PROXY_TOKEN"), "{stderr}");
    assert!(
        stderr.contains("HEADROOM_ALLOW_UNAUTHENTICATED_BIND"),
        "{stderr}"
    );
}

#[tokio::test]
async fn loopback_listen_without_token_still_starts() {
    let mut child = proxy("127.0.0.1:0").spawn().expect("spawn proxy");
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        child.try_wait().expect("try_wait").is_none(),
        "loopback bind must start without a token"
    );
    let _ = child.kill().await;
}

#[test]
fn default_listen_address_is_loopback() {
    use clap::Parser;
    use headroom_proxy::config::CliArgs;
    let args = CliArgs::try_parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:9"])
        .expect("parse");
    if std::env::var_os("HEADROOM_PROXY_LISTEN").is_none() {
        assert!(args.listen.ip().is_loopback(), "{}", args.listen);
    }
}
