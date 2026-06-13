# Modal deployments

The two deep-learning models used by Storybear, deployed on
[Modal](https://modal.com) as **remote HTTP services** (not Modal-SDK workloads).

| Script | Model | Role | Endpoint style |
|--------|-------|------|----------------|
| `minicpm_v.py` | `openbmb/MiniCPM-V-4.6` | Image-to-text (captions, ranking, editorial copy) | vLLM **OpenAI-compatible** `/v1/chat/completions` |
| `minicpm_v_llamacpp.py` | `openbmb/MiniCPM-V-4.6-gguf` | Same, as a llama.cpp alternative for cold-start/cost comparison | llama.cpp **OpenAI-compatible** `/v1/chat/completions` |
| `flux_klein.py` | `black-forest-labs/FLUX.2-klein-base-4B` | Image-to-image (artistic post-processing) | FastAPI `/edit` (returns PNG) |

The VLM is an LLM, so it gets a standard OpenAI-compatible API. FLUX is a
diffusion model with no chat-completions analog, so it gets a dedicated
image-editing endpoint.

## Prerequisites

```bash
pip install modal
modal setup        # one-time authentication
```

FLUX.2 weights are gated on Hugging Face, so create a Modal secret with a token
that has access:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

The first deploy downloads each model into a shared persistent Volume
(`storybear-hf-cache`); later cold starts reuse it.

## Deploy

```bash
modal deploy modal/minicpm_v.py            # vLLM VLM service URL
modal deploy modal/minicpm_v_llamacpp.py   # llama.cpp VLM service URL (comparison)
modal deploy modal/flux_klein.py           # FLUX service URL
```

Both VLM deployments expose the same OpenAI-compatible API, so the curl /
OpenAI-client calls below work against either URL. Compare cold starts via:

```bash
modal app logs storybear-minicpm-v            # vLLM
modal app logs storybear-minicpm-v-llamacpp   # llama.cpp
```

Each deploy prints a public URL, e.g.
`https://<workspace>--storybear-minicpm-v-serve.modal.run`.

## Use the VLM (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<workspace>--storybear-minicpm-v-serve.modal.run/v1",
    api_key="EMPTY",  # vLLM does not require a key unless you add --api-key
)

# Text-only
client.chat.completions.create(
    model="minicpm-v",
    messages=[{"role": "user", "content": "Why are histograms useful in EDA?"}],
)

# With an image (base64 data URL, OpenAI vision format)
client.chat.completions.create(
    model="minicpm-v",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Caption this chart."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
        ],
    }],
)
```

or using curl:

```sh
# Text-only
curl https://<workspace>--storybear-minicpm-v-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-v",
    "messages": [{"role": "user", "content": "Why are histograms useful in EDA?"}]
  }'

## Use FLUX (image edit)

```bash
curl -X POST https://<workspace>--storybear-flux-klein-fluxklein-web.modal.run/edit \
    -F "prompt=Add bold black contours, lazy meme-like doodle style." \
    -F "image=@chart.png" \
    -o edited.png
```

## Wiring into Storybear

The pipeline accepts injectable inference functions
(`captionist_it2t_func`, `foodie_it2t_func`, `editor_it2t_func`,
`artist_i2i_func`). Thin HTTP wrappers route those calls to the deployed
services:

```python
import base64
import requests
from openai import OpenAI

VLM_URL = "https://<workspace>--storybear-minicpm-v-serve.modal.run/v1"
# Class-based asgi_app: the subdomain includes the class name (fluxklein).
FLUX_URL = "https://<workspace>--storybear-flux-klein-fluxklein-web.modal.run/edit"

vlm = OpenAI(base_url=VLM_URL, api_key="EMPTY")


def it2t_summary_func(system_prompt, record_list):
    messages = [{"role": "system", "content": system_prompt}]
    for r in record_list:
        content = r.caption if hasattr(r, "caption") else r
        messages.append({"role": "user", "content": content})
    resp = vlm.chat.completions.create(model="minicpm-v", messages=messages)
    return resp.choices[0].message.content


def flux_i2i_func(prompt, file_path):
    with open(file_path, "rb") as f:
        resp = requests.post(FLUX_URL, data={"prompt": prompt}, files={"image": f})
    resp.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(resp.content)
    return file_path
```

To send chart images to the VLM (not just captions), build the user content as
an OpenAI vision message with a base64 `image_url` from `record.plot_path`.

## Notes

- Both services default to a single `A100` GPU. Change `GPU_TYPE` in each script
  (e.g. `"L40S"`, `"A100-80GB"`, `"H100"`) to trade cost for throughput.
- vLLM batches concurrent requests; `@modal.concurrent(max_inputs=32)` lets one
  warm container serve the whole pipeline. `scaledown_window=300` keeps it warm
  for five minutes between calls.
- Pin `vllm` in `minicpm_v.py` to a release that supports MiniCPM-V-4.6 for
  reproducible builds.
- To require auth on the vLLM endpoint, add `--api-key <key>` to the `vllm serve`
  command and pass the same key as the OpenAI client's `api_key`.
