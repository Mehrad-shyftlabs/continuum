"""Regression test for the Redis session provider's TLS/SSL wiring.

Bug: ``RedisSessionProvider.initialize`` enabled TLS by adding ``ssl=True`` to the
kwargs forwarded to ``(Blocking)ConnectionPool``. redis-py selects the TLS
transport via the connection *class* (``SSLConnection``), not an ``ssl`` kwarg —
so ``ssl=True`` is forwarded verbatim to ``AbstractConnection.__init__`` and the
pool raises ``TypeError: ... unexpected keyword argument 'ssl'`` the moment it
builds its first connection (lazily, at command time). Every session read/write
against a TLS Redis (e.g. AWS ElastiCache in-transit encryption) therefore failed.

These are pure unit tests: ``make_connection`` instantiates the connection object
without opening a socket, so no live Redis is required to prove the fix.
"""

from __future__ import annotations

from redis.asyncio import Connection, SSLConnection

from continuum.session.config import SessionConfig
from continuum.session.providers.redis import RedisSessionProvider


def _ssl_config(**overrides) -> SessionConfig:
    base = dict(
        enabled=True,
        redis_host="localhost",
        redis_port=6390,
        redis_ssl=True,
    )
    base.update(overrides)
    return SessionConfig(**base)


def test_ssl_pool_builds_an_ssl_connection_without_error() -> None:
    """redis_ssl=True must yield a pool whose connections are SSLConnection.

    On the buggy code this raises ``TypeError: AbstractConnection.__init__()
    got an unexpected keyword argument 'ssl'`` from make_connection().
    """
    provider = RedisSessionProvider(config=_ssl_config(), auto_initialize=True)
    assert provider._pool is not None, "pool should have been created for redis_ssl=True"

    conn = provider._pool.make_connection()
    assert isinstance(conn, SSLConnection)


def test_ssl_pool_does_not_forward_a_bare_ssl_kwarg() -> None:
    """The invalid ``ssl`` kwarg must never reach the connection kwargs."""
    provider = RedisSessionProvider(config=_ssl_config(), auto_initialize=True)
    assert "ssl" not in provider._pool.connection_kwargs


def test_non_ssl_pool_still_uses_plain_connection() -> None:
    """redis_ssl=False must keep the default (plaintext) connection class."""
    provider = RedisSessionProvider(
        config=_ssl_config(redis_ssl=False), auto_initialize=True
    )
    conn = provider._pool.make_connection()
    assert isinstance(conn, Connection)
    assert not isinstance(conn, SSLConnection)


def test_ssl_cert_reqs_is_passed_through_when_configured() -> None:
    """An explicit ssl_cert_reqs must reach the SSLConnection (needed for
    self-signed / custom-CA Redis such as a local TLS test container)."""
    provider = RedisSessionProvider(
        config=_ssl_config(redis_ssl_cert_reqs="none"), auto_initialize=True
    )
    conn = provider._pool.make_connection()
    assert isinstance(conn, SSLConnection)
    # redis-py normalises the string to an ssl.VerifyMode; CERT_NONE == 0.
    import ssl as _ssl

    assert conn.cert_reqs == _ssl.CERT_NONE
