"""
Modal deployment for the MiniCPM-V-4.6 vision-language model, served through
vLLM's OpenAI-compatible HTTP server.

This exposes the image-to-text (it2t) model used by Storybear's Captionist,
Foodie and Editor stages as a standard `/v1/chat/completions` endpoint, so it
can be called from any OpenAI client (Python, curl, other languages) instead of
the Modal SDK.

Deploy:
    modal deploy modal/minicpm_v.py

The deploy prints a public URL like:
    https://<workspace>--storybear-minicpm-v-serve.modal.run

Use it as an OpenAI base URL (note the trailing /v1):
    from openai import OpenAI
    client = OpenAI(base_url="https://...modal.run/v1", api_key="EMPTY")
    client.chat.completions.create(model="minicpm-v", messages=[...])
"""

import modal

MODEL_NAME = "openbmb/MiniCPM-V-4.6"  # or "openbmb/MiniCPM-V-4.6-Thinking"

# Stable model id that OpenAI clients pass as `model=...`.
SERVED_MODEL_NAME = "minicpm-v"

# GPU type used for inference. MiniCPM-V-4.6 fits on a single A100.
GPU_TYPE = "A100"
N_GPU = 1

# Port the vLLM server listens on inside the container.
VLLM_PORT = 8000

# Where Hugging Face weights are cached inside the container, backed by a Volume.
CACHE_DIR = "/cache"

app = modal.App("storybear-minicpm-v")

# Persistent cache for model weights, shared across cold starts and revisions.
hf_cache = modal.Volume.from_name("storybear-hf-cache", create_if_missing=True)


def download_model():
    """Pre-download the model weights into the cache volume at build time."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} into the cache volume...")
    snapshot_download(MODEL_NAME)
    print("Download complete.")


# Container image with vLLM and the Hugging Face download accelerator.
# NOTE: pin `vllm` to a release that supports MiniCPM-V-4.6 for reproducibility.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            # Fast Xet-based downloads (replaces the deprecated HF_HUB_ENABLE_HF_TRANSFER).
            "HF_XET_HIGH_PERFORMANCE": "1",
            # The pip vLLM wheel ships only the CUDA runtime, not the toolkit
            # (nvcc). FlashInfer's sampler would try to JIT-compile a CUDA kernel
            # at startup and crash with "Could not find nvcc"; fall back to
            # vLLM's native PyTorch sampler, which needs no compiler.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    # Bake the weights into the cache volume so cold starts skip the download.
    .run_function(download_model, volumes={CACHE_DIR: hf_cache})
)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={CACHE_DIR: hf_cache},
    # Keep a warm container for five minutes between calls so a multi-stage
    # pipeline run does not reload the model for every request.
    scaledown_window=300,
    timeout=600,
)
# vLLM batches concurrent requests, so let one container handle many at once.
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    """Launch the vLLM OpenAI-compatible API server."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # MiniCPM-V ships custom modelling code.
        "--trust-remote-code",
        # Skip torch.compile + CUDA-graph capture (~80s + warmup here) for much
        # faster cold starts and a smaller startup surface. Trades some
        # steady-state throughput; drop this once cold start is acceptable.
        "--enforce-eager",
        # Cap context to keep KV-cache memory in check on a single GPU.
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        # Allow several images per prompt (Foodie compares pairs of charts).
        # Recent vLLM expects a JSON value here, not key=value.
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    print("Starting vLLM server:", " ".join(cmd))
    # web_server waits for the port to come up; do not block here.
    subprocess.Popen(cmd)
