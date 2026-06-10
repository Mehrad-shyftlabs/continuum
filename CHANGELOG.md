# Changelog

All notable changes to Continuum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Continuum adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub issue templates (`bug.yml`, `feature.yml`, `question.yml`) and a chooser config that disables blank issues and routes security reports to a private advisory.
- Pull request template with Conventional-Commits typing, DCO checkbox, and lint/type/test gates.
- `CODEOWNERS` for automatic review routing.
- Dependabot vulnerability **alerts** (surfaced in the Security tab); automated dependency PRs are disabled.
- `MAINTAINERS.md` with the current maintainer list, tone rules, and escalation path.
- `SECURITY.md` with the private disclosure channel and severity SLAs.
- Minimal CI workflow: ruff lint + format check and unit tests on `main`/`dev`.
- `continuum` CLI for one-command infra startup — `continuum up [minimal|standard|full]`, plus `down`, `status`, `logs`, and `config-path`. The Docker Compose stack and Temporal dynamic config are now bundled in the wheel, so there's no compose file to locate or copy after a `pip install`. Each profile writes a managed block to `./.env` so the SDK only targets services that are actually running.
- All published Docker host ports are overridable via `.env` (e.g. `QDRANT_PORT`, `SESSION_REDIS_PORT`, `MILVUS_PORT`, `LANGFUSE_WEB_PORT`, `TEMPORAL_PORT`), with defaults preserving prior behavior — avoids collisions on multi-project machines.
- "Releasing (maintainers)" section in `CONTRIBUTING.md` linking the canonical [`docs/versioning.md`](docs/versioning.md) publish guide.

### Changed
- _Nothing yet._

### Fixed
- Docker healthchecks for `qdrant` (now probes `/readyz` over bash `/dev/tcp`, since the image ships no `curl`) and `temporal` (`BIND_ON_IP=0.0.0.0` so the localhost healthcheck can reach the frontend) — both previously reported `unhealthy` while serving correctly.
- `continuum down`/`status`/`logs` now activate all compose profiles, so profiled containers from `minimal`/`standard` are no longer orphaned.

---

## [0.2.2] — 2026-06

### Fixed
- `continuum.__version__` is now derived from the installed package metadata via
  `importlib.metadata` instead of a hardcoded string, so it always matches the
  distribution version. In `0.2.1` the attribute incorrectly reported `0.2.0`
  because the literal was never bumped. See [docs/versioning.md](docs/versioning.md).

---

## [0.2.1] — 2026-06

### Changed
- Renamed the importable package from `orchestrator` to `continuum`. The
  distribution is unchanged (`pip install shyftlabs-continuum`); imports are now
  `import continuum` / `from continuum.… import …`. Runtime config keys
  (Temporal task queue, memory collection defaults, session key prefix,
  Prometheus metric names) and the `initialize_orchestrator`/`shutdown_orchestrator`
  functions are unchanged.

### Added
- `continuum/py.typed` marker so consumers receive the package's type hints (PEP 561).

### Deprecated
- _Nothing yet._

### Removed
- _Nothing yet._

### Fixed
- _Nothing yet._

### Security
- _Nothing yet._

---

## [0.2.0] — 2026-05

Initial public release. See the repository history for details prior to this changelog being introduced.

[Unreleased]: https://github.com/shyftlabs/continuum/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/shyftlabs/continuum/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/shyftlabs/continuum/releases/tag/v0.2.0
