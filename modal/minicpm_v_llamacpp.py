"""
Alternative Modal deployment for MiniCPM-V-4.6, served with llama.cpp instead
of vLLM, for a cold-start / cost comparison.

llama.cpp skips vLLM's memory-profiling pass, CUDA-graph capture and
torch.compile, so it typically boots in seconds. With a quantized GGUF it also
loads far less data and fits a cheaper GPU. Steady-state throughput is lower
than vLLM, so this is the better choice when reports are bursty/one-off and
cold-start latency dominates.

It exposes the same OpenAI-compatible `/v1/chat/completions` API as the vLLM
deployment, so the exact same curl / OpenAI-client calls work against it.

Deploy:
    modal deploy modal/minicpm_v_llamacpp.py

Compare cold starts by grepping each app's logs between "loading model" and the
"server is listening" line:
    modal app logs storybear-minicpm-v            # vLLM
    modal app logs storybear-minicpm-v-llamacpp   # llama.cpp
"""

import modal

# GGUF weights for MiniCPM-V-4.6. This repo holds both the main model GGUF and
# the multimodal projector (mmproj) needed for vision.
GGUF_REPO = "openbmb/MiniCPM-V-4.6-gguf"

# Preferred quantization for the main model file. A 4-bit quant loads fast and
# fits a small GPU; switch to e.g. "F16" to match vLLM's precision.
PREFERRED_QUANT = "Q4_K_M"

# Stable model id that OpenAI clients pass as `model=...`.
SERVED_MODEL_NAME = "minicpm-v"

# A quantized GGUF fits a cheap GPU. Use "A100" to match the vLLM deployment for
# an apples-to-apples cold-start comparison; "A10G" is enough for a 4-bit quant.
GPU_TYPE = "L40S"

# Port llama-server listens on inside the container.
LLAMA_PORT = 8000

# Path to the server binary inside the official llama.cpp CUDA image. Adjust if
# a future image revision changes the layout (check with the image's shell).
LLAMA_SERVER_BIN = "/app/llama-server"

# Where Hugging Face weights are cached inside the container, backed by a Volume.
CACHE_DIR = "/cache"

app = modal.App("storybear-minicpm-v-llamacpp")

# Persistent cache for model weights, shared across cold starts and revisions.
hf_cache = modal.Volume.from_name("storybear-hf-cache", create_if_missing=True)


def resolve_gguf_files() -> tuple[str, str]:
    """
    Locate the main model GGUF and the mmproj GGUF in the repo, downloading them
    into the cache volume if missing. Returns (model_path, mmproj_path).

    Filenames are detected rather than hard-coded, since GGUF repos vary in how
    they name quant variants and the projector file.
    """
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(GGUF_REPO) if f.endswith(".gguf")]

    # The vision projector is the file whose name contains "mmproj".
    mmproj_file = next(f for f in files if "mmproj" in f.lower())

    # The model is the preferred quant among the remaining GGUFs, else the first.
    candidates = [f for f in files if "mmproj" not in f.lower()]
    model_file = next(
        (f for f in candidates if PREFERRED_QUANT.lower() in f.lower()),
        candidates[0],
    )

    print(f"Using model file: {model_file}")
    print(f"Using mmproj file: {mmproj_file}")
    model_path = hf_hub_download(GGUF_REPO, model_file)
    mmproj_path = hf_hub_download(GGUF_REPO, mmproj_file)
    return model_path, mmproj_path


def download_model():
    """Pre-download the GGUF + mmproj into the cache volume at build time."""
    print(f"Downloading GGUF weights from {GGUF_REPO} into the cache volume...")
    resolve_gguf_files()
    print("Download complete.")


# Official prebuilt llama.cpp CUDA server image — no compilation needed. We add
# a Python so huggingface_hub can fetch the GGUFs at build/start time.
llama_image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.11",
    )
    # The image's ENTRYPOINT is `llama-server`; clear it so Modal can run Python
    # (for the download step) and so we invoke the binary explicitly in serve().
    .entrypoint([])
    .pip_install("huggingface_hub[hf_xet]")
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    # Bake the weights into the cache volume so cold starts skip the download.
    .run_function(download_model, volumes={CACHE_DIR: hf_cache})
)


@app.function(
    image=llama_image,
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: hf_cache},
    # Keep a warm container for five minutes between calls.
    scaledown_window=300,
    timeout=600,
)
# llama-server batches a handful of requests with continuous batching below.
@modal.concurrent(max_inputs=4)
@modal.web_server(port=LLAMA_PORT, startup_timeout=600)
def serve():
    """Launch the llama.cpp OpenAI-compatible server with vision enabled."""
    import subprocess

    model_path, mmproj_path = resolve_gguf_files()

    cmd = [
        LLAMA_SERVER_BIN,
        "-m",
        model_path,
        # Multimodal projector — required for image input.
        "--mmproj",
        mmproj_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(LLAMA_PORT),
        # Offload all transformer layers to the GPU.
        "-ngl",
        "99",
        # Context length.
        "-c",
        "8192",
        # Continuous batching + a few parallel slots for the pipeline's bursts.
        "-cb",
        "--parallel",
        "4",
        # Model name reported by the OpenAI-compatible API.
        "--alias",
        SERVED_MODEL_NAME,
    ]
    print("Starting llama.cpp server:", " ".join(cmd))
    # web_server waits for the port to come up; do not block here.
    subprocess.Popen(cmd)
