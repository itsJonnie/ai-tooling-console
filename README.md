# AI Tooling Console - Jonathan Sher

AI Tooling Console is a lightweight internal-style web application designed to support machine learning engineering workflows. It focuses on **launching runs, monitoring execution, debugging failures, and observing system health** — mirroring the kinds of tooling used in production AI environments.

This project is intentionally scoped to demonstrate **fullstack engineering, reliability thinking, and ML workflow support**, rather than model complexity.

---

## Why This Exists

Modern AI teams rely on internal tooling to:
- Launch and track experiments
- Debug failures quickly
- Inspect logs and metrics in real time
- Maintain system reliability as scale increases

AI Tooling Console explores how such tools can be designed with:
- Clear workflows
- Strong observability
- Production-oriented engineering practices

---

## Core Capabilities

### Run Management
- Create training runs with configurable parameters
- Track run lifecycle: `QUEUED → RUNNING → SUCCEEDED / FAILED`
- Store run configuration and metadata

### Observability & Debugging
- Structured logs emitted during execution
- Time-series metrics (e.g. loss, accuracy)
- Explicit failure states with error messages and root-cause annotations

### System Health
- Health check endpoint for service monitoring
- System-level metrics for run throughput and failure rates

---

## Architecture Overview


The system is designed to evolve incrementally:
- Phase 1: In-memory state for fast iteration
- Phase 2: Persistent storage (Postgres)
- Phase 3: Containerized + Kubernetes deployment

---

## Tech Stack

**Frontend**
- React
- TypeScript
- Next.js

**Backend**
- Python
- FastAPI
- Async background tasks

**Infrastructure**
- Docker (local development)
- Kubernetes manifests (deployment reference)

---

## API (Initial MVP)

- `POST /runs` – create a run
- `GET /runs` – list runs
- `GET /runs/{id}` – run details
- `GET /runs/{id}/logs` – execution logs
- `GET /runs/{id}/metrics` – metrics data
- `GET /health` – service health

---

## Kubernetes (Reference)

This repository includes minimal Kubernetes manifests demonstrating:
- Separate frontend and backend deployments
- Health probes for reliability
- Environment-based configuration

These manifests are intended to show **deployment readiness and operational thinking**, not full production hardening.

---

## Design Principles

- **Tooling over modeling** – focus is on workflows, not ML novelty
- **Debuggability first** – failures should be explainable, not opaque
- **Production mindset** – health checks, metrics, and clear state transitions
- **Incremental complexity** – start simple, evolve toward scale

---

## Future Improvements

- Persistent database (Postgres)
- Authentication and role-based access
- Run comparison and experiment versioning
- Streaming logs and metrics (SSE/WebSockets)
- Integration with real ML training pipelines

---

## Status

🚧 Active development — starting with a minimal MVP and iterating toward a production-style internal tool.
