"""E2E correctness test: verify PegaFlow produces identical outputs to baseline vLLM.

This is the "canary" test - the single most important test for PegaFlow.
It verifies the core contract:

    Given the same prompt + model + sampling params,
    output must be identical with or without PegaFlow.

Usage:
    # Requires pegaflow-server running:
    # cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --device 0 --pool-size 30gb

    cd python && pytest tests/test_e2e_correctness.py -v -s

    # Or with a specific model:
    pytest tests/test_e2e_correctness.py -v -s --model /path/to/model
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="module")
def model(request) -> str:
    """Get model from command line or use default."""
    return request.config.getoption("--model")


@pytest.fixture(scope="module")
def base_port(request) -> int:
    """Get base port from command line."""
    return request.config.getoption("--e2e-port")


class VLLMServer:
    """Context manager for vLLM server lifecycle.

    Reused from examples/bench_kv_cache.py with minimal modifications.
    """

    def __init__(
        self,
        model: str,
        port: int,
        use_pegaflow: bool = False,
        log_file: Path | None = None,
        max_model_len: int | None = None,
    ):
        self.model = model
        self.port = port
        self.use_pegaflow = use_pegaflow
        self.log_file = log_file
        self.max_model_len = max_model_len
        self.health_endpoints = ["/health", "/v1/models"]
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    def __enter__(self):
        """Start the vLLM server."""
        env = os.environ.copy()

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--block-size",
            "64",
            "--no-enable-prefix-caching",
        ]

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if self.use_pegaflow:
            kv_config = {
                "kv_connector": "PegaKVConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "pegaflow.connector",
            }
            cmd.extend(["--kv-transfer-config", json.dumps(kv_config)])

        server_label = "PegaFlow" if self.use_pegaflow else "Baseline"
        print(f"\n[{server_label}] Starting vLLM server on port {self.port}")

        if self.log_file:
            print(f"[{server_label}] Logging to: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

        self._wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the vLLM server."""
        if self.process:
            server_label = "PegaFlow" if self.use_pegaflow else "Baseline"
            print(f"\n[{server_label}] Stopping vLLM server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
            print("Server stopped.\n")

        if self.log_handle:
            self.log_handle.close()

    def _wait_for_ready(self, timeout: int = 180):
        """Wait for the server to be ready to accept requests."""
        start_time = time.time()
        print("Waiting for server to be ready...")
        while time.time() - start_time < timeout:
            for endpoint in self.health_endpoints:
                url = f"http://localhost:{self.port}{endpoint}"
                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        print(f"Server is ready! (checked {endpoint})\n")
                        time.sleep(2)  # Extra buffer
                        return
                except requests.exceptions.RequestException:
                    continue
            time.sleep(2)

        raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def call_openai_api(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    seed: int = 42,
) -> dict:
    """Call vLLM's OpenAI-compatible API.

    Returns dict with 'text' and 'logprobs' (if available).
    """
    url = f"http://localhost:{port}/v1/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "logprobs": 5,  # Request logprobs for detailed comparison
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    choice = data["choices"][0]
    return {
        "text": choice["text"],
        "logprobs": choice.get("logprobs"),
        "finish_reason": choice["finish_reason"],
        "usage": data.get("usage"),
    }


class TestE2ECorrectness:
    """E2E correctness tests for PegaFlow.

    These tests verify that PegaFlow produces identical outputs to baseline vLLM.
    """

    @pytest.fixture(scope="class")
    def log_dir(self, tmp_path_factory) -> Path:
        """Create a temporary directory for server logs."""
        return tmp_path_factory.mktemp("vllm_logs")

    def test_cache_roundtrip_correctness(self, model: str, base_port: int, log_dir: Path):
        """Core contract: baseline, cold cache, and warm cache must produce identical outputs.

        This is THE canary test. If this fails, something is fundamentally broken.
        """
        # Test prompts - varied lengths to test different code paths
        test_prompts = [
            # Short prompt
            "The capital of France is",
            # Medium prompt with more context
            "In computer science, a hash table is a data structure that implements an associative array. "
            "The main advantage of hash tables is",
            # Longer prompt to test block boundaries
            "Machine learning is a subset of artificial intelligence that focuses on building systems "
            "that can learn from data. Deep learning, a subset of machine learning, uses neural networks "
            "with many layers. The key benefit of deep learning is",
        ]

        results = {"baseline": [], "pegaflow_cold": [], "pegaflow_warm": []}

        # Phase 1: Baseline vLLM (no PegaFlow)
        print("\n" + "=" * 60)
        print("PHASE 1: BASELINE vLLM (no KV connector)")
        print("=" * 60)

        baseline_log = log_dir / "baseline.log"
        with VLLMServer(model, base_port, use_pegaflow=False, log_file=baseline_log):
            for prompt in test_prompts:
                result = call_openai_api(base_port, model, prompt)
                results["baseline"].append(result)
                print(f"Baseline output: {result['text'][:50]}...")

        # Phase 2: PegaFlow vLLM
        print("\n" + "=" * 60)
        print("PHASE 2: PegaFlow vLLM (cold + warm cache)")
        print("=" * 60)

        pegaflow_log = log_dir / "pegaflow.log"
        with VLLMServer(model, base_port + 1, use_pegaflow=True, log_file=pegaflow_log):
            # Cold cache run
            print("\n--- Cold Cache Run ---")
            for prompt in test_prompts:
                result = call_openai_api(base_port + 1, model, prompt)
                results["pegaflow_cold"].append(result)
                print(f"Cold output: {result['text'][:50]}...")

            # Warm cache run (same prompts again)
            print("\n--- Warm Cache Run ---")
            for prompt in test_prompts:
                result = call_openai_api(base_port + 1, model, prompt)
                results["pegaflow_warm"].append(result)
                print(f"Warm output: {result['text'][:50]}...")

        # Phase 3: Verify correctness
        print("\n" + "=" * 60)
        print("PHASE 3: VERIFICATION")
        print("=" * 60)

        for i, prompt in enumerate(test_prompts):
            baseline_text = results["baseline"][i]["text"]
            cold_text = results["pegaflow_cold"][i]["text"]
            warm_text = results["pegaflow_warm"][i]["text"]

            print(f"\nPrompt {i + 1}: {prompt[:40]}...")
            print(f"  Baseline: {baseline_text[:40]}...")
            print(f"  Cold:     {cold_text[:40]}...")
            print(f"  Warm:     {warm_text[:40]}...")

            # Core assertions
            assert cold_text == warm_text, (
                f"Prompt {i + 1}: Cold and warm cache outputs differ!\n"
                f"Cold: {cold_text}\n"
                f"Warm: {warm_text}"
            )

            assert baseline_text == cold_text, (
                f"Prompt {i + 1}: Baseline and PegaFlow outputs differ!\n"
                f"Baseline: {baseline_text}\n"
                f"PegaFlow: {cold_text}"
            )

            print("  [PASS] All outputs match!")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED - Core contract verified!")
        print("=" * 60)


class TestE2ECacheHit:
    """Test that warm cache actually hits (performance sanity check)."""

    @pytest.fixture(scope="class")
    def log_dir(self, tmp_path_factory) -> Path:
        """Create a temporary directory for server logs."""
        return tmp_path_factory.mktemp("vllm_logs_cache")

    def test_warm_cache_faster_than_cold(self, model: str, base_port: int, log_dir: Path):
        """Verify warm cache provides measurable speedup (sanity check).

        This is not a strict test - just verifies caching is working at all.
        """
        # Use a longer prompt to make cache benefit more visible
        prompt = (
            "The history of artificial intelligence began in antiquity, with myths, "
            "stories and rumors of artificial beings endowed with intelligence. "
            "The seeds of modern AI were planted by philosophers who attempted to "
            "describe the process of human thinking as the mechanical manipulation "
            "of symbols. This work culminated in the invention of the programmable "
            "digital computer in the 1940s. The field of AI research was founded at "
            "a workshop at Dartmouth College in 1956. The answer is"
        )

        pegaflow_log = log_dir / "pegaflow_timing.log"
        with VLLMServer(model, base_port + 2, use_pegaflow=True, log_file=pegaflow_log):
            # Cold run with timing
            cold_start = time.time()
            cold_result = call_openai_api(base_port + 2, model, prompt, max_tokens=10)
            cold_time = time.time() - cold_start

            # Warm run with timing
            warm_start = time.time()
            warm_result = call_openai_api(base_port + 2, model, prompt, max_tokens=10)
            warm_time = time.time() - warm_start

            print(f"\nCold cache time: {cold_time:.3f}s")
            print(f"Warm cache time: {warm_time:.3f}s")
            print(f"Speedup: {cold_time / warm_time:.2f}x")

            # Outputs must match
            assert cold_result["text"] == warm_result["text"], (
                f"Cold and warm outputs differ!\n"
                f"Cold: {cold_result['text']}\n"
                f"Warm: {warm_result['text']}"
            )

            # Warm should generally be faster (allow some variance)
            # This is a soft check - mainly for sanity
            if warm_time >= cold_time:
                print(
                    f"WARNING: Warm cache ({warm_time:.3f}s) not faster than cold ({cold_time:.3f}s). "
                    "This might be OK for short prompts or first-time model loading."
                )
