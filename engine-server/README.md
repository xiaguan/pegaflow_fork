# PegaFlow Engine gRPC Server

This crate wraps the Rust `PegaEngine` and exposes the same functionality as
`python/pegaflow/engine_server.py`, but over a tonic gRPC service.

## Building

The binary embeds CPython via PyO3 so it can reconstruct CUDA IPC tensors with
Torch, just like the Python server. Before running cargo commands, point PyO3 to
the exact interpreter you want (usually the repo's `.venv`) so linking works and
the runtime can import `pegaflow.ipc_wrapper`:

```bash
export PYO3_PYTHON="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd)/python:$(pwd)/.venv/lib/python3.10/site-packages"
cargo run -p engine-server -- --addr 0.0.0.0:50055 --device 0
```

Adjust the Python path if your venv uses a different minor version.

## Flags

- `--addr`: Bind address for the tonic server (`127.0.0.1:50055` by default).
- `--device`: Default CUDA device id. This matches the Python server's behavior
  and ensures Torch/CUDA are initialized on the correct GPU.
