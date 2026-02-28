# Docker Compose Setup

The Temporal integration adds three services to the project's
`docker-compose.yml`. This guide covers their configuration and usage.

## Services overview

| Service | Image | Port | Purpose |
|---|---|---|---|
| `postgres-temporal` | `postgres:17` | `127.0.0.1:5434` | Dedicated Postgres for Temporal persistence |
| `temporal` | `temporalio/auto-setup:latest` | `7233` | Temporal server with auto-setup |
| `temporal-ui` | `temporalio/ui:latest` | `8233` | Web UI for workflow monitoring |

These services are on the `orchestrator-network` bridge and are separate from
the Langfuse Postgres instance.

## Starting Temporal services

Start only the Temporal stack:

```bash
docker compose up -d postgres-temporal temporal temporal-ui
```

Start everything (including Langfuse, Redis, Qdrant):

```bash
docker compose up -d
```

## Verifying health

```bash
# Check all service statuses
docker compose ps

# Temporal server health
docker compose exec temporal temporal operator namespace describe --namespace default

# Postgres health
docker compose exec postgres-temporal pg_isready -U temporal
```

## Temporal server configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DB` | `postgres12` | Database type |
| `DB_PORT` | `5432` | Database port (internal) |
| `POSTGRES_USER` | `temporal` | Postgres username |
| `POSTGRES_PWD` | `temporal` | Postgres password |
| `POSTGRES_SEEDS` | `postgres-temporal` | Postgres hostname |
| `DYNAMIC_CONFIG_FILE_PATH` | `config/dynamicconfig/development-sql.yaml` | Dynamic config file |

### Dynamic configuration

The file `temporal/dynamicconfig/development-sql.yaml` is mounted into the
Temporal container. Default content for development:

```yaml
system.forceSearchAttributesCacheRefreshOnRead:
  - value: true
    constraints: {}
```

You can add Temporal dynamic config overrides here. Changes require a container
restart:

```bash
docker compose restart temporal
```

## Temporal UI

Access the Temporal Web UI at **http://localhost:8233**.

Features:
- View workflow executions and their history
- Inspect workflow input/output
- Send signals to running workflows
- Query workflow state
- View task queue metrics

### UI environment variables

| Variable | Default | Description |
|---|---|---|
| `TEMPORAL_ADDRESS` | `temporal:7233` | Temporal server address |
| `TEMPORAL_CORS_ORIGINS` | `http://localhost:3000` | CORS allowed origins |

## Postgres (Temporal)

A dedicated Postgres instance for Temporal's persistence layer.

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_USER` | `temporal` | Username |
| `POSTGRES_PASSWORD` | `temporal` | Password |
| `POSTGRES_DB` | `temporal` | Database name |

Data is persisted in the `temporal_postgres_data` Docker volume.

## Volumes

| Volume | Service | Purpose |
|---|---|---|
| `temporal_postgres_data` | `postgres-temporal` | Temporal database storage |

## Network

All Temporal services are on the `orchestrator-network` bridge network,
shared with the SDK's Redis and Qdrant services.

## Production considerations

For production deployments, consider:

1. **Use a managed Temporal service** (Temporal Cloud) instead of self-hosting.
2. **Secure Postgres credentials** -- replace the default `temporal/temporal`
   with strong passwords.
3. **Enable TLS** between the SDK and Temporal server.
4. **Separate the Temporal database** from other application databases.
5. **Configure resource limits** for the Temporal server container.
6. **Set up monitoring** with Temporal's Prometheus metrics endpoint.

### Connecting to Temporal Cloud

```dotenv
TEMPORAL_HOST=your-namespace.tmprl.cloud:7233
TEMPORAL_NAMESPACE=your-namespace
```

You'll also need to configure mTLS certificates. See the
[Temporal Cloud documentation](https://docs.temporal.io/cloud) for details.

## Stopping services

```bash
# Stop Temporal services only
docker compose stop temporal temporal-ui postgres-temporal

# Stop and remove everything (data persisted in volumes)
docker compose down

# Stop and remove everything including volumes (data lost)
docker compose down -v
```

## Troubleshooting

### Temporal server won't start

Check that `postgres-temporal` is healthy first:

```bash
docker compose logs postgres-temporal
```

The Temporal `auto-setup` image runs database migrations on startup. If Postgres
isn't ready, Temporal will fail. The `depends_on` + health check should handle
this, but network issues can cause delays.

### "Connection refused" from the SDK

Ensure `TEMPORAL_HOST` matches the exposed port:
- From the host machine: `localhost:7233`
- From another Docker container on the same network: `temporal:7233`

### Temporal UI shows no workflows

Verify the namespace matches. The SDK defaults to `"default"`, and the
`auto-setup` image creates this namespace automatically.

## Next steps

- [Getting Started](getting-started.md) -- full setup walkthrough
- [Custom Workflows](custom-workflows.md) -- writing your own workflows
