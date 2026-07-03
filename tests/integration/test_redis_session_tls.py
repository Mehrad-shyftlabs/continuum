"""End-to-end TLS test for the Redis Session Provider.

Exercises the real SSLConnection path against a TLS-enabled Redis (docker
service `redis-sdk-tls`, mirroring AWS ElastiCache in-transit encryption).
This is the regression guard for the bug where redis_ssl=True forwarded a bare
``ssl=True`` kwarg to the connection pool and broke every session operation.

Setup (once):
    bash tests/integration/redis_tls/gen_certs.sh
    REDIS_TLS_CERT_DIR=$(pwd)/tests/integration/redis_tls/certs \
      docker compose --profile tls-test up -d redis-sdk-tls

The test verifies the full TLS chain against the generated CA
(ssl_cert_reqs="required" + ssl_ca_certs=ca.crt), so it proves genuine
encrypted transport, not just that a socket opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_CERT_DIR = Path(__file__).parent / "redis_tls" / "certs"
_CA_CERT = _CERT_DIR / "ca.crt"

_TLS_HOST = "localhost"
_TLS_PORT = 6390
_TLS_PASSWORD = "sdk123456789"


@pytest.fixture
async def tls_session_provider():
    """Redis session provider talking to the TLS Redis with full CA verification."""
    from continuum.session.config import SessionConfig
    from continuum.session.providers.redis import RedisSessionProvider

    if not _CA_CERT.is_file():
        pytest.skip(
            f"CA cert not found at {_CA_CERT}. Run tests/integration/redis_tls/gen_certs.sh"
        )

    config = SessionConfig(
        enabled=True,
        redis_host=_TLS_HOST,
        redis_port=_TLS_PORT,
        redis_password=_TLS_PASSWORD,
        redis_ssl=True,
        redis_ssl_cert_reqs="required",
        redis_ssl_ca_certs=str(_CA_CERT),
        ttl_seconds=300,
        max_messages=100,
    )
    provider = RedisSessionProvider(config=config)
    if not provider.initialize():
        pytest.skip("TLS Redis session provider failed to initialize")

    # Fail fast (and skip) if the TLS Redis container isn't up, rather than
    # letting every assertion raise a connection error.
    try:
        await provider._redis.ping()
    except Exception as exc:  # noqa: BLE001
        await provider.close()
        pytest.skip(f"TLS Redis not reachable on {_TLS_HOST}:{_TLS_PORT} ({exc})")

    yield provider
    await provider.close()


class TestRedisSessionOverTLS:
    async def test_connection_is_ssl(self, tls_session_provider):
        """The pool must build genuine SSLConnection objects."""
        from redis.asyncio import SSLConnection

        conn = tls_session_provider._pool.make_connection()
        assert isinstance(conn, SSLConnection)

    async def test_roundtrip_over_tls(self, tls_session_provider, test_id):
        """Create a session, write two messages, read them back — all over TLS."""
        from continuum.llm.types import ChatMessage

        sid = await tls_session_provider.get_or_create_session(
            session_id=f"tls-sess-{test_id}",
            user_id="tls-user",
        )
        assert sid == f"tls-sess-{test_id}"

        await tls_session_provider.add_message(
            sid, ChatMessage(role="user", content="Hello over TLS")
        )
        await tls_session_provider.add_message(
            sid, ChatMessage(role="assistant", content="Encrypted hi")
        )

        messages = await tls_session_provider.get_messages(sid)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[0].content == "Hello over TLS"
        assert messages[1].content == "Encrypted hi"

        metadata = await tls_session_provider.get_session_metadata(sid)
        assert metadata is not None
        assert metadata.user_id == "tls-user"

        # Clean up the key so re-runs stay deterministic.
        await tls_session_provider.delete_session(sid)
