#!/usr/bin/env bash
# Generate a self-signed CA + Redis server certificate for the local TLS Redis
# integration test (test_redis_session_tls.py / docker-compose `redis-sdk-tls`).
#
# Output (written next to this script, git-ignored):
#   certs/ca.crt      — CA cert the client trusts (SESSION_REDIS_SSL_CA_CERTS)
#   certs/redis.crt   — server cert (signed by the CA), SAN=localhost,127.0.0.1
#   certs/redis.key   — server private key
#
# Certs are NOT committed — regenerate on demand. Safe to re-run (idempotent).
set -euo pipefail

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/certs"
mkdir -p "$CERT_DIR"

if [[ -f "$CERT_DIR/redis.crt" && "${FORCE:-0}" != "1" ]]; then
  echo "certs already present in $CERT_DIR (set FORCE=1 to regenerate)"
  exit 0
fi

echo "Generating self-signed CA + Redis server cert in $CERT_DIR ..."

# 1. CA — must carry basicConstraints:CA + keyUsage:keyCertSign, otherwise
#    OpenSSL 3.x rejects it with "CA cert does not include key usage extension".
openssl genrsa -out "$CERT_DIR/ca.key" 4096
openssl req -x509 -new -nodes -sha256 -days 3650 \
  -key "$CERT_DIR/ca.key" \
  -subj "/O=continuum-test/CN=continuum-test-ca" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -out "$CERT_DIR/ca.crt"

# 2. Server key + CSR
openssl genrsa -out "$CERT_DIR/redis.key" 2048
openssl req -new -sha256 \
  -key "$CERT_DIR/redis.key" \
  -subj "/O=continuum-test/CN=localhost" \
  -out "$CERT_DIR/redis.csr"

# 3. Sign the server cert with the CA (SAN so hostname verification also passes)
openssl x509 -req -sha256 -days 3650 \
  -in "$CERT_DIR/redis.csr" \
  -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" -CAcreateserial \
  -extfile <(printf "subjectAltName=DNS:localhost,IP:127.0.0.1\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth") \
  -out "$CERT_DIR/redis.crt"

rm -f "$CERT_DIR/redis.csr" "$CERT_DIR/ca.srl"
chmod 644 "$CERT_DIR"/*.crt "$CERT_DIR"/*.key
echo "Done."
