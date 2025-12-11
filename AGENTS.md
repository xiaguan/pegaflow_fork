# PegaFlow Agents Documentation

## Project Overview

PegaFlow is a Rust-based key-value storage engine with Python bindings. The project provides:
1. **PegaEngine**: High-performance KV storage implemented in Rust with Python bindings via PyO3
2. **PegaKVConnector**: vLLM KV connector for distributed inference with KV cache transfer

## Project Structure

```
pegaflow/
├── Cargo.toml                 # Workspace configuration
├── AGENTS.md                  # This file - project documentation
├── README.md                  # Project overview
├── assets/                    # Logos and images
├── docs/                      # Additional docs (metrics, P/D setup)
├── examples/                  # Python examples and benchmarks
│   ├── basic_vllm.py
│   ├── bench_kv_cache.py
│   ├── run_vllm_pd_with_pega.py
│   └── metric/                # Grafana/OTel compose configs
├── pegaflow-core/             # Core Rust library (GPU-aware KV engine)
│   ├── Cargo.toml
│   └── src/                   # Storage, transfer, pinned pool, metrics
├── pegaflow-server/           # gRPC server + router binary
│   ├── Cargo.toml
│   ├── build.rs               # Prost/tonic codegen
│   ├── proto/engine.proto     # Engine RPC definition
│   └── src/                   # service, registry, proto glue, router bin
└── python/                    # Python package (published as "pegaflow")
    ├── Cargo.toml             # Rust crate config for PyO3
    ├── pyproject.toml         # Python package config
    ├── README.md              # Python package documentation
    ├── src/lib.rs             # PyO3 bindings for PegaEngine + RPC client
    └── pegaflow/              # Python source code
        ├── __init__.py        # Package entry point
        ├── connector/         # vLLM v1 connector (scheduler/worker)
        ├── ipc_wrapper.py     # CUDA IPC tensor wrapper
        └── logging_utils.py   # Connector logging helpers
```

## Architecture

### pegaflow-core

Core Rust library implementing the GPU-aware KV engine (pinned pool, block storage, async load/save, metrics).

### pegaflow-server

Tonic-based gRPC server wrapping `PegaEngine` plus a lightweight P/D router binary for disaggregated setups.

### python/

The Python package layer (published as `pegaflow` on PyPI):

#### Rust Extension (src/lib.rs)
- Uses PyO3 to expose `PegaEngine` and the gRPC client for Python
- Built as `pegaflow.pegaflow` extension module using maturin

#### Python Package (pegaflow/)
- **`__init__.py`**: Package entry point, exports `PegaEngine` and `PegaKVConnector`
- **`connector/`**: vLLM v1 KV connector (scheduler/worker split, IPC handling)
- **`ipc_wrapper.py`**: CUDA IPC tensor wrapper used during registration
- **`logging_utils.py`**: Logging/timing helpers for the connector

## Development Workflow

### Building the Project

1. **Build Rust crates**:
   ```bash
   cargo build
   ```

2. **Build Python bindings**:
   ```bash
   cd python
   maturin develop
   ```

3. **Run Python examples**:
   ```bash
   python examples/basic_usage.py
   python examples/basic_vllm.py
   ```

### Testing

The `examples/basic_usage.py` script demonstrates basic usage:
- Create a PegaEngine instance
- Put key-value pairs
- Get values by key
- Remove keys
- Verify operations

The `examples/basic_vllm.py` script demonstrates vLLM integration:
- Configure vLLM to use PegaKVConnector
- Start the gRPC engine server (see README) and load a model with KV transfer enabled
- Generate text twice to exercise cold/warm KV paths

## Agent Workflow

When working on this project, agents should:

1. **Understand the separation of concerns**:
   - `pegaflow-core`: Pure Rust logic, no Python dependencies
   - `pegaflow-server`: gRPC server and router wrapping `pegaflow-core`
   - `python/src/`: Python bindings and RPC client only, delegates to `pegaflow-core`/server
   - `python/pegaflow/`: Pure Python utilities (vLLM connector, logging, IPC)

2. **Make changes in the correct layer**:
   - Business logic changes → `pegaflow-core`
   - Python API changes → `python/src/lib.rs`
   - vLLM connector changes → `python/pegaflow/connector/`

3. **Test changes**:
   - Run `cargo test` for Rust tests
   - Run `cd python && maturin develop` to rebuild Python bindings
   - Run `python examples/basic_usage.py` to verify PegaEngine
   - Run `python examples/basic_vllm.py` to verify vLLM integration

4. **Follow Rust best practices**:
   - Use proper error handling
   - Follow ownership and borrowing rules
   - Prefer `NonNull` over `*mut` in unsafe code
   - Write idiomatic Rust code

5. **Follow Python style (3.9+)**:
   - Use native generics (`list`, `dict`, `set`, `tuple`) instead of `typing.List`, `typing.Dict`, etc.
   - Use PEP 604 union syntax (`X | Y`, `X | None`) instead of `typing.Union`, `typing.Optional`
   - Logging: use `%s` formatting (`logger.info("x=%s", x)`) instead of f-strings to avoid evaluation overhead

6. **Follow PyO3 patterns**:
   - Use `#[pyclass]` for Python-exposed structs
   - Use `#[pymethods]` for Python-exposed methods
   - Handle Python exceptions properly

## Dependencies

### Rust Dependencies
- `pyo3`: Python bindings framework (python/ only)

### Python Dependencies
- `maturin`: Build tool for PyO3 projects
- `vllm`: Large language model inference framework (for connector integration)

### Build Tools
- `uv`: Fast Python package installer and environment manager

## Package Naming

The Python package is published as **`pegaflow`** on PyPI:
- Package name: `pegaflow` (all lowercase, no hyphens)
- Import name: `from pegaflow import PegaEngine, PegaKVConnector`
- Version: `0.0.1` (initial development release)

PyPI allows lowercase package names, and this follows Python naming conventions (PEP 8).

## Future Enhancements

Potential areas for expansion:
- Implement PegaKVConnector methods for actual KV cache transfer
- Persistence (save/load from disk)
- Advanced data structures (sorted sets, lists)
- Async operations
- Transaction support
- Compression
- Replication
