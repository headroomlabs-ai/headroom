from __future__ import annotations

import asyncio
import json
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import websockets

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.egress import AuthorizedDestination
from headroom.proxy.gateway.transport import http_client, tls_context, websocket_connection


@pytest.fixture
def local_pki(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Gateway test CA")])
    now = datetime.now(timezone.utc)
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "llm.internal.example")]))
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("llm.internal.example"), x509.DNSName("api.anthropic.com")]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    ca_file, leaf_file, key_file = (
        tmp_path / filename for filename in ("ca.pem", "leaf.pem", "key.pem")
    )
    ca_file.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    leaf_file.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(leaf_file, key_file)
    raw = json.loads(
        (
            Path(__file__).parents[2]
            / "docs/proposals/unified-api-gateway/examples/gateway.private-upstream.json"
        ).read_text()
    )
    raw["transport"]["ca_bundle"] = str(ca_file)
    return context, raw


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["trusted", "untrusted", "wrong-hostname"])
async def test_http_custom_ca_and_hostname_verification_on_real_tls(local_pki, mode):
    context, raw = local_pki
    received = []

    async def serve(reader, writer):
        try:
            received.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=context)
    port = server.sockets[0].getsockname()[1]
    host = "wrong.internal.example" if mode == "wrong-hostname" else "llm.internal.example"
    target = f"https://{host}:{port}/v1/responses"
    if mode == "untrusted":
        raw.pop("transport")
    # Pinning/TLS is tested independently of policy: production denies loopback addresses.
    destination = AuthorizedDestination(host, port, ("127.0.0.1",), target)
    async with server, http_client(GatewayConfigSnapshot.model_validate(raw)) as client:
        if mode == "trusted":
            response = await client.get(target, extensions={"gateway_destination": destination})
            assert response.content == b"ok"
            assert f"Host: {host}:{port}".encode() in received[0]
        else:
            with pytest.raises(httpx.ConnectError):
                await client.get(target, extensions={"gateway_destination": destination})
            assert received == []


@pytest.mark.asyncio
async def test_websocket_custom_ca_on_real_tls(local_pki):
    context, raw = local_pki

    async def serve(connection):
        await connection.send(await connection.recv())

    async with websockets.serve(serve, "127.0.0.1", 0, ssl=context) as server:
        port = server.sockets[0].getsockname()[1]
        target = f"https://llm.internal.example:{port}/v1/responses"
        destination = AuthorizedDestination("llm.internal.example", port, ("127.0.0.1",), target)
        async with websocket_connection(
            destination, {}, tls_context(GatewayConfigSnapshot.model_validate(raw))
        ) as client:
            await client.send("hello")
            assert await client.recv() == "hello"
