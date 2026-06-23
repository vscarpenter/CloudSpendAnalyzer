"""Shared pytest fixtures and test isolation.

These fixtures keep the suite hermetic and fast:

* ``_isolate_from_network`` blocks outbound network connections so a test that
  escapes its mocks fails fast with a clear error instead of hanging on a live
  LLM/AWS endpoint (the suite previously hung for minutes on a real Gemini call).
* The same fixture no-ops ``time.sleep`` so exponential-backoff retry paths do
  not slow the suite to a crawl.
"""

import os
import socket
import sys
import time

import pytest

_real_sleep = time.sleep


def _smart_sleep(seconds=0, *args, **kwargs):
    """Skip backoff sleeps inside app/library code (keeps the suite fast) but
    honor sleeps called directly from test code (e.g. cache TTL-expiry tests)."""
    caller_file = sys._getframe(1).f_code.co_filename
    if f"{os.sep}tests{os.sep}" in caller_file:
        return _real_sleep(seconds)
    return None

# Force gRPC (used by the Gemini SDK) to resolve DNS via the system resolver
# instead of its own C-core resolver, so the getaddrinfo block below applies.
# Must be set before grpc's C extension initializes, i.e. at conftest import.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

# Loopback hosts a test may legitimately need (e.g. a local mock server).
_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", ""}

_real_socket_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _is_local(host) -> bool:
    return host in _ALLOWED_HOSTS or (isinstance(host, str) and host.startswith("127."))


@pytest.fixture(autouse=True)
def _isolate_from_network(monkeypatch):
    """Block non-local network and neutralize backoff sleeps for every test.

    A test that escapes its mocks fails fast with a clear error instead of
    hanging on a live LLM/AWS endpoint. Blocking both ``connect`` and
    ``getaddrinfo`` (DNS) catches plain sockets and gRPC alike.
    """

    def _guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, (tuple, list)) else address
        if not _is_local(host):
            raise RuntimeError(
                f"Blocked network connection to {host!r} during tests. "
                "Mock the provider/AWS client instead of making a live call."
            )
        return _real_socket_connect(self, address, *args, **kwargs)

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            raise socket.gaierror(f"Blocked DNS lookup for {host!r} during tests")
        return _real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    monkeypatch.setattr(time, "sleep", _smart_sleep)
    socket.setdefaulttimeout(2)
    yield
    socket.setdefaulttimeout(None)
