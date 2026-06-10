"""
Modal deployment for the FLUX.2-klein image-to-image model, served as an HTTP
endpoint.

This exposes the image-to-image (i2i) model used by Storybear's Artist stage as
a FastAPI service: POST a reference image plus a style prompt, get the edited
image (PNG) back. Diffusion models have no chat-completions analog, so this is a
dedicated endpoint rather than an OpenAI-compatible one.

FLUX.2 weights are gated on Hugging Face. Create a Modal secret named
`huggingface-secret` exposing `HF_TOKEN` before deploying:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx

Deploy:
    modal deploy modal/flux_klein.py

The deploy prints a public URL like (the class name appears in the subdomain
because this is an asgi_app on a class):
    https://<workspace>--storybear-flux-klein-fluxklein-web.modal.run

Call it (multipart form: `image` file + `prompt` field):
    curl -X POST https://...modal.run/edit \
        -F "prompt=Add bold black contours, lazy meme-like doodle style." \
        -F "image=@chart.png" \
        -o edited.png
"""

import modal

MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-4B"

# FLUX.2-klein is a 4B model; a single A100 handles it with CPU offload enabled.
GPU_TYPE = "A100"

# Where Hugging Face weights are cached inside the container, backed by a Volume.
CACHE_DIR = "/cache"

app = modal.App("storybear-flux-klein")

# Persistent cache for model weights, shared across cold starts and revisions.
hf_cache = modal.Volume.from_name("storybear-hf-cache", create_if_missing=True)

# Hugging Face token secret, required because FLUX.2 weights are gated.
hf_secret = modal.Secret.from_name("huggingface-secret")


def download_model():
    """Pre-download the model weights into the cache volume at build time."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} into the cache volume...")
    snapshot_download(MODEL_NAME)
    print("Download complete.")


# Container image. Flux2KleinPipeline ships in recent diffusers releases; we
# install from the development branch to guarantee the pipeline is available.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # `git` is required to install diffusers from the development branch below.
    .apt_install("git")
    .pip_install(
        "torch>=2.11",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=5.6,<6",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "pillow",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
        "python-multipart",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Bake the weights into the cache volume so cold starts skip the download.
    .run_function(download_model, volumes={CACHE_DIR: hf_cache}, secrets=[hf_secret])
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: hf_cache},
    secrets=[hf_secret],
    # Keep a warm container for five minutes between calls so a batch of charts
    # does not pay the cold-start cost per image.
    scaledown_window=300,
    timeout=600,
)
class FluxKlein:
    @modal.enter()
    def load(self):
        """Load the diffusion pipeline once per container, on the GPU."""
        import torch
        from diffusers import Flux2KleinPipeline

        print(f"Loading {MODEL_NAME}...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        # Offload idle submodules to CPU to keep peak VRAM in check.
        self.pipe.enable_model_cpu_offload()
        print("Pipeline ready.")

    def _run(
        self,
        prompt: str,
        image_bytes: bytes,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> bytes:
        """Apply the prompt's style to the image and return PNG bytes."""
        import io
        from PIL import Image

        reference = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = self.pipe(
            prompt=prompt,
            image=reference,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue()

    @modal.asgi_app()
    def web(self):
        """FastAPI app exposing the image-editing endpoint."""
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import Response

        web_app = FastAPI(title="Storybear FLUX.2-klein image editor")

        @web_app.post("/edit")
        async def edit(
            prompt: str = Form(...),
            image: UploadFile = File(...),
            num_inference_steps: int = Form(4),
            guidance_scale: float = Form(4.0),
        ):
            """Edit `image` according to `prompt` and return the PNG result."""
            image_bytes = await image.read()
            png = self._run(prompt, image_bytes, num_inference_steps, guidance_scale)
            return Response(content=png, media_type="image/png")

        return web_app









