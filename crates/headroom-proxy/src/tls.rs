//! Upstream TLS trust for the Rust proxy.
//!
//! Upstreams are verified against Mozilla's bundled roots plus the operating
//! system store, which is where IT installs a corporate TLS-inspection root
//! (Zscaler, Netskope, ...). Deployments without such a store (containers,
//! CI) hand the root over as a PEM file instead; these env vars add it,
//! matching the Python proxy's policy:
//!
//! * `HEADROOM_CA_BUNDLE` — Headroom's own knob.
//! * `NODE_EXTRA_CA_CERTS` — what Claude Code users already set.
//!
//! Both are additive. `SSL_CERT_FILE` / `SSL_CERT_DIR` are honored by
//! `rustls-native-certs` itself, as a replacement for the OS store.
//!
//! HTTP upstreams (reqwest) get the extras via [`extra_root_certificates`];
//! WebSocket upstreams get a complete rustls config from
//! [`websocket_tls_config`].

use std::path::Path;
use std::sync::{Arc, OnceLock};

use rustls_pki_types::pem::PemObject;
use rustls_pki_types::CertificateDer;

/// Env vars naming additive PEM bundles, in the order they are loaded.
pub const ADDITIVE_CA_VARS: [&str; 2] = ["HEADROOM_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"];

/// Root certificates from the additive env vars, for reqwest. Missing or
/// unparsable files are logged and skipped: a bad path must not take the
/// proxy down.
pub fn extra_root_certificates() -> Vec<reqwest::Certificate> {
    extra_root_ders()
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
        .collect()
}

/// Install ring as the process-wide rustls [`CryptoProvider`].
///
/// This binary links two providers — ring (reqwest, tokio-tungstenite) and
/// aws-lc-rs (the AWS SDK behind `aws-config`). rustls refuses to guess
/// between them: `ClientConfig::builder()` panics unless a default has been
/// installed, and reqwest reaches for that default when it builds its TLS
/// client. Installing ring explicitly makes the choice deliberate and keeps
/// every TLS client in the process on one provider — the same one
/// [`websocket_tls_config`] names. Idempotent: a second call (tests, embedded
/// use) is a no-op.
///
/// ring rather than aws-lc-rs because it is the provider reqwest already
/// defaults to, needs no C toolchain or cmake at build time, and the AWS SDK
/// carries its own explicit configuration either way.
///
/// [`CryptoProvider`]: rustls::crypto::CryptoProvider
pub fn install_process_crypto_provider() {
    // `Err` means a provider is already installed, which is exactly the
    // state we want.
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// rustls client config for `wss://` upstreams: webpki + OS + additive roots,
/// on an explicit crypto provider. Built once per process.
pub fn websocket_tls_config() -> Arc<rustls::ClientConfig> {
    static CONFIG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| Arc::new(build_websocket_tls_config(&extra_root_ders())))
        .clone()
}

fn build_websocket_tls_config(extra: &[CertificateDer<'static>]) -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let native = rustls_native_certs::load_native_certs();
    if !native.errors.is_empty() {
        tracing::warn!(
            event = "tls_native_roots_errors",
            errors = ?native.errors,
            "some OS root certificates could not be loaded"
        );
    }
    roots.add_parsable_certificates(native.certs);
    roots.add_parsable_certificates(extra.iter().cloned());

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("ring supports the default TLS versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    // A WebSocket upgrade is an HTTP/1.1 request.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    config
}

fn extra_root_ders() -> Vec<CertificateDer<'static>> {
    extra_root_ders_from(|var| std::env::var(var).ok())
}

fn extra_root_ders_from(lookup: impl Fn(&str) -> Option<String>) -> Vec<CertificateDer<'static>> {
    let mut certs = Vec::new();
    for var in ADDITIVE_CA_VARS {
        let Some(path) = lookup(var).filter(|p| !p.is_empty()) else {
            continue;
        };
        match load_pem_bundle(Path::new(&path)) {
            Ok(found) => {
                tracing::info!(
                    event = "tls_ca_bundle_loaded",
                    env_var = var,
                    path = %path,
                    certificates = found.len(),
                    "loaded extra upstream root certificates"
                );
                certs.extend(found);
            }
            Err(err) => {
                tracing::warn!(
                    event = "tls_ca_bundle_skipped",
                    env_var = var,
                    path = %path,
                    error = %err,
                    "could not load extra root certificates; skipping"
                );
            }
        }
    }
    certs
}

fn load_pem_bundle(path: &Path) -> Result<Vec<CertificateDer<'static>>, String> {
    let certs = CertificateDer::pem_file_iter(path)
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if certs.is_empty() {
        return Err("no PEM certificates found".to_string());
    }
    Ok(certs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Throwaway self-signed CA, only parsed here, never used for a handshake.
    const TEST_CA: &str = "-----BEGIN CERTIFICATE-----
MIIBuTCCAV+gAwIBAgIUQJOjUYts91bSLsemwb5EuzdkOnMwCgYIKoZIzj0EAwIw
MjEVMBMGA1UECgwMWnNjYWxlciBJbmMuMRkwFwYDVQQDDBBIZWFkcm9vbSBUZXN0
IENBMB4XDTI2MDkyODE0MjM1MloXDTM2MDkyNTE0MjM1MlowMjEVMBMGA1UECgwM
WnNjYWxlciBJbmMuMRkwFwYDVQQDDBBIZWFkcm9vbSBUZXN0IENBMFkwEwYHKoZI
zj0CAQYIKoZIzj0DAQcDQgAEKZ0h9e4jj/eJiBVh4eMZ3d+pcugxj/hEhqzoAo9r
+7KL4dGgqrg2GH4IP6sfKNdssKHshDcjz+HWiKVGyCx08KNTMFEwHQYDVR0OBBYE
FPFmGy3B6aRYxGMB1P/up4Y05PxbMB8GA1UdIwQYMBaAFPFmGy3B6aRYxGMB1P/u
p4Y05PxbMA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDSAAwRQIgZTHC/ZR0
SI8i1dYBBaT8a2e/CVF6MmgXObYWEg2YYk8CIQC8e2+jUZDkYV+Ct3y0JQO7OSfp
+Yyjbym7JFsfsp4fVQ==
-----END CERTIFICATE-----
";

    fn write_temp(name: &str, contents: &str) -> String {
        let dir = std::env::temp_dir().join(format!("headroom-tls-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path.display().to_string()
    }

    #[test]
    fn unset_and_empty_vars_add_nothing() {
        assert!(extra_root_ders_from(|_| None).is_empty());
        assert!(extra_root_ders_from(|_| Some(String::new())).is_empty());
    }

    #[test]
    fn missing_file_is_skipped() {
        let certs = extra_root_ders_from(|var| {
            (var == "HEADROOM_CA_BUNDLE").then(|| "/nonexistent/headroom-ca.pem".to_string())
        });
        assert!(certs.is_empty());
    }

    #[test]
    fn non_pem_file_is_skipped() {
        let p = write_temp("not-a-cert.pem", "hello");
        let certs = extra_root_ders_from(|var| (var == "NODE_EXTRA_CA_CERTS").then(|| p.clone()));
        assert!(certs.is_empty());
    }

    #[test]
    fn both_additive_vars_are_loaded() {
        let p = write_temp("corp-root.pem", TEST_CA);
        let certs = extra_root_ders_from(|_| Some(p.clone()));
        assert_eq!(certs.len(), ADDITIVE_CA_VARS.len());
        assert!(reqwest::Certificate::from_der(certs[0].as_ref()).is_ok());
    }

    #[test]
    fn websocket_config_builds_without_a_process_default_provider() {
        // rustls' own `ClientConfig::builder()` panics in this binary (two
        // providers compiled in); ours must not, and must trust the extras.
        let p = write_temp("corp-root-ws.pem", TEST_CA);
        let extra = extra_root_ders_from(|var| (var == "HEADROOM_CA_BUNDLE").then(|| p.clone()));
        let config = build_websocket_tls_config(&extra);
        assert_eq!(config.alpn_protocols, vec![b"http/1.1".to_vec()]);
    }
}
