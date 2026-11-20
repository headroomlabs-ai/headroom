"""Root-CA key/cert consistency across concurrent wraps and interrupted writes."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from headroom.proxy import agy_ca as ca


def test_concurrent_initialization_keeps_one_matching_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ca, "_RSA_KEY_BITS", 2048)
    pairs = [ca._generate_root_ca(), ca._generate_root_ca()]
    first_written, second_written = threading.Event(), threading.Event()
    generation_lock = threading.Lock()
    generated = []
    original_write = ca._write_secure

    def generate():
        with generation_lock:
            pair = pairs[len(generated)]
            generated.append(pair)
            return pair

    first_pem = pairs[0][0].private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )

    def interleaved_write(path, data):
        if path.name != ca._CA_KEY_NAME:
            return original_write(path, data)
        if data == first_pem:
            original_write(path, data)
            first_written.set()
            second_written.wait(0.3)
        else:
            assert first_written.wait(2)
            original_write(path, data)
            second_written.set()

    monkeypatch.setattr(ca, "_generate_root_ca", generate)
    monkeypatch.setattr(ca, "_write_secure", interleaved_write)
    start = threading.Barrier(2)

    def initialize():
        start.wait(timeout=2)
        return ca.ensure_root_ca(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(initialize) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]
    disk_key = serialization.load_pem_private_key(results[0][2].read_bytes(), password=None)
    disk_cert = x509.load_pem_x509_certificate(results[0][3].read_bytes())
    assert _public_bytes(disk_key.public_key()) == _public_bytes(disk_cert.public_key())
    assert results[0][1].serial_number == results[1][1].serial_number
    assert len(generated) == 1


def _public_bytes(key):
    return key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def test_existing_mismatched_pair_is_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ca, "_RSA_KEY_BITS", 2048)
    key, _, key_path, cert_path = ca.ensure_root_ca(tmp_path)
    _, wrong_cert = ca._generate_root_ca()
    ca._write_secure(cert_path, wrong_cert.public_bytes(serialization.Encoding.PEM))
    repaired_key, repaired_cert, _, _ = ca.ensure_root_ca(tmp_path)
    assert _public_bytes(repaired_key.public_key()) == _public_bytes(repaired_cert.public_key())
    assert _public_bytes(repaired_key.public_key()) != _public_bytes(key.public_key())
    assert repaired_cert.serial_number != wrong_cert.serial_number
    assert key_path.exists()


def test_separate_processes_share_one_ca(tmp_path):
    worker = """
import json, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from cryptography.hazmat.primitives import hashes, serialization
import headroom.proxy.agy_ca as ca
base, ready, go = map(pathlib.Path, sys.argv[2:])
ready.write_text('ready')
deadline = time.monotonic() + 15
while not go.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('CA worker start deadline')
    time.sleep(0.01)
generate = ca._generate_root_ca
def delayed_generation():
    time.sleep(0.2)
    return generate()
ca._generate_root_ca = delayed_generation
key, cert, *_ = ca.ensure_root_ca(base)
def public_bytes(value):
    return value.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
print(json.dumps({'fingerprint': cert.fingerprint(hashes.SHA256()).hex(),
                  'matches': public_bytes(key.public_key()) == public_bytes(cert.public_key())}))
"""
    root = Path(__file__).resolve().parents[1]
    go = tmp_path / "go"
    ready_files = [tmp_path / f"ready-{i}" for i in range(3)]
    children = []
    try:
        for ready in ready_files:
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        worker,
                        str(root),
                        str(tmp_path / "ca-state"),
                        str(ready),
                        str(go),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 15
        while not all(path.exists() for path in ready_files):
            assert time.monotonic() < deadline, "CA workers did not become ready"
            time.sleep(0.01)
        go.write_text("go")
        results = []
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            results.append(json.loads(stdout))
        assert all(result["matches"] for result in results)
        assert len({result["fingerprint"] for result in results}) == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
