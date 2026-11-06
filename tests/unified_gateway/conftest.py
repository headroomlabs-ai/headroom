"""Deterministic DNS for unit tests whose upstream transport is already a fake."""

import socket

import pytest


@pytest.fixture(autouse=True)
def gateway_unit_dns(monkeypatch):
    real_getaddrinfo = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        # Preserve loopback/process harness resolution; production never sees this fixture.
        if host in {
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
            "us-central1-aiplatform.googleapis.com",
            "bedrock-runtime.us-east-1.amazonaws.com",
        }:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
