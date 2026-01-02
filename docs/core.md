# Core Module

Foundational infrastructure: container, lifecycle, health checks, and configuration.

## Overview

- **Container**: Dependency injection container
- **Lifecycle**: Resource lifecycle management
- **Health Checks**: Service health verification
- **Configuration**: Global settings management

## Container (Dependency Injection)

```python
from orchestrator.core import Container, ContainerConfig, get_container

# Get global container
container = get_container()

# Access clients
llm = container.llm_client
memory = container.memory_client
session = container.session_client

# Custom container
config = ContainerConfig(
    enable_memory=True,
    enable_session=True,
    enable_langfuse=True,
)
container = Container(config=config)
```

## Lifecycle Management

```python
from orchestrator.core import (
    initialize_orchestrator,
    shutdown_orchestrator,
    get_lifecycle_manager,
)

# Initialize
lifecycle = initialize_orchestrator()

# Shutdown
await shutdown_orchestrator()

# Or use lifecycle manager
manager = get_lifecycle_manager()
await manager.shutdown()
```

## Health Checks

```python
from orchestrator.core import get_health_checker, check_all_health

# Check all services
checker = get_health_checker()
result = await checker.check_all()

# Check specific service
redis_status = await checker.check_redis()
qdrant_status = await checker.check_qdrant()
langfuse_status = await checker.check_langfuse()
llm_status = await checker.check_llm()

# Or use convenience function
result = await check_all_health()
```

## Configuration

Global configuration via environment variables:

```python
from orchestrator.config import settings

# Access settings
model = settings.default_llm_model
memory_enabled = settings.memory_enabled
session_enabled = settings.session_enabled
```

### Key Settings

- `DEFAULT_LLM_MODEL`: Default LLM model
- `MEMORY_ENABLED`: Enable/disable memory
- `SESSION_ENABLED`: Enable/disable sessions
- `LANGFUSE_ENABLED`: Enable/disable Langfuse
- `QDRANT_HOST`: Qdrant host
- `SESSION_REDIS_HOST`: Redis host for sessions

## Health Check Script

```bash
# Check all services
python scripts/health_check.py

# Check specific service
python scripts/health_check.py --service redis

# JSON output
python scripts/health_check.py --json
```

## Types

- `Container`: DI container
- `ContainerConfig`: Container configuration
- `HealthCheck`: Health check interface
- `HealthCheckResult`: Health check result
- `HealthStatus`: Health status enum
- `OrchestratorLifecycle`: Lifecycle manager

## Configuration Validation

```python
from orchestrator.core import validate_configuration

# Validate configuration
errors = validate_configuration()
if errors:
    print("Configuration errors:", errors)
```
