"""
Remote inference backends for the Storybear pipeline.

Drop-in replacement for ``storybear.local_inference``: exposes the same
``it2t_summary_func`` and ``flux_i2i_func`` callables with the same signatures,
but routes them to the Modal-hosted services under ``modal/`` instead of loading
the models locally. Swap the import and the pipeline works unchanged::

    # from storybear.local_inference import it2t_summary_func, flux_i2i_func
    from storybear.remote_inference import it2t_summary_func, flux_i2i_func

Unlike the local version, the image-to-text path also attaches each chart image
to the request (OpenAI vision format), so the model actually sees the plot.

Configuration (environment variables)
-------------------------------------
STORYBEAR_VLM_URL
    OpenAI-compatible VLM base URL, e.g.
    ``https://<workspace>--storybear-minicpm-v-serve.modal.run``
    (a trailing ``/v1`` is appended if missing). Works against either the vLLM
    or the llama.cpp deployment.
STORYBEAR_FLUX_URL
    FLUX image-edit endpoint, e.g.
    ``https://<workspace>--storybear-flux-klein-web.modal.run/edit``
STORYBEAR_VLM_MODEL
    Model name passed to the API (default ``minicpm-v``).
STORYBEAR_VLM_API_KEY
    API key if the endpoint requires one (default ``EMPTY``).
STORYBEAR_MODAL_KEY / STORYBEAR_MODAL_SECRET
    Proxy auth token id/secret for the Modal endpoints (created at
    https://modal.com/settings/proxy-auth-tokens). When both are set they are
    sent as the ``Modal-Key``/``Modal-Secret`` headers on every request; when
    unset no auth headers are sent, so unauthenticated endpoints still work.
STORYBEAR_HTTP_TIMEOUT
    Per-request timeout in seconds (default ``600``); generous to absorb cold
    starts.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import List, Union

from storybear.data_structures import PlotRecord
from storybear.utils import klein_size
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "minicpm-v"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 600.0

# Foodie's pairwise comparisons only need a single label, so we downscale the
# chart images to cut vision tokens. Tune via STORYBEAR_COMPARE_IMAGE_PX.
DEFAULT_COMPARE_IMAGE_PX = 512

# How long to keep retrying a request while the service is still warming up
# (e.g. llama.cpp returns 503 "Loading model" during cold start). Tune via
# STORYBEAR_RETRY_SECONDS.
DEFAULT_RETRY_SECONDS = 180.0


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _timeout() -> float:
    return float(os.environ.get("STORYBEAR_HTTP_TIMEOUT", DEFAULT_TIMEOUT))


def _retry_seconds() -> float:
    return float(os.environ.get("STORYBEAR_RETRY_SECONDS", DEFAULT_RETRY_SECONDS))


def _vlm_base_url() -> str:
    url = os.environ.get("STORYBEAR_VLM_URL")
    if not url:
        raise RuntimeError(
            "STORYBEAR_VLM_URL is not set. Point it at the deployed VLM service, "
            "e.g. https://<workspace>--storybear-minicpm-v-serve.modal.run"
        )
    url = url.rstrip("/")
    # The OpenAI client expects a base URL ending in /v1.
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _proxy_auth_headers() -> dict:
    """
    Headers carrying the Modal proxy auth token, or an empty dict when no token
    is configured. Both id and secret must be present; a partial config is
    treated as no auth so a misconfigured half-token does not silently send a
    useless header.
    """
    key = os.environ.get("STORYBEAR_MODAL_KEY")
    secret = os.environ.get("STORYBEAR_MODAL_SECRET")
    if key and secret:
        return {"Modal-Key": key, "Modal-Secret": secret}
    return {}


def _flux_url() -> str:
    url = os.environ.get("STORYBEAR_FLUX_URL")
    if not url:
        raise RuntimeError(
            "STORYBEAR_FLUX_URL is not set. Point it at the deployed FLUX endpoint, "
            "e.g. https://<workspace>--storybear-flux-klein-fluxklein-web.modal.run/edit"
        )
    # The FastAPI app only serves POST /edit; tolerate a base URL without it.
    if not url.rstrip("/").endswith("/edit"):
        url = url.rstrip("/") + "/edit"
    return url


# Lazily built OpenAI client, cached for reuse across pipeline stages.
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=_vlm_base_url(),
            api_key=os.environ.get("STORYBEAR_VLM_API_KEY", DEFAULT_API_KEY),
            timeout=_timeout(),
            # We own retries in _create_with_retry (patient, model-load aware).
            max_retries=0,
            # Modal proxy auth headers (empty dict when no token is configured).
            default_headers=_proxy_auth_headers(),
        )
    return _client


# Status codes worth retrying: model still loading / overloaded / transient 5xx.
_RETRYABLE_STATUS = (429, 500, 502, 503)


def _create_with_retry(**kwargs):
    """
    Call chat.completions.create, retrying transient failures (notably the
    503 "Loading model" a cold-starting server returns) until the model is up
    or the retry window elapses.
    """
    import time

    from openai import APIConnectionError, APIStatusError

    client = _get_client()
    deadline = time.monotonic() + _retry_seconds()
    delay = 1.0
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.chat.completions.create(**kwargs)
        except APIConnectionError as exc:
            last_exc = exc
        except APIStatusError as exc:
            if exc.status_code not in _RETRYABLE_STATUS:
                raise
            last_exc = exc
        if time.monotonic() >= deadline:
            raise last_exc
        logger.info("VLM not ready (attempt %d), retrying in %.0fs...", attempt, delay)
        time.sleep(delay)
        delay = min(delay * 2, 15.0)


# ---------------------------------------------------------------------------
# Image-to-text (VLM)
# ---------------------------------------------------------------------------

def _image_data_url(path: Path, max_px: int | None = None) -> str:
    """
    Encode an image file as an OpenAI-style base64 data URL.

    When ``max_px`` is set the image is downscaled (preserving aspect ratio) so
    its longest side is at most ``max_px`` pixels, which sharply reduces the
    number of vision tokens sent to the model.
    """
    if max_px:
        import io

        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((max_px, max_px))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{data}"

    suffix = path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in ("jpg", "jpeg") else (suffix or "png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{media_type};base64,{data}"


def _record_to_message(
    record: Union[PlotRecord, str],
    image_max_px: int | None = None,
) -> dict:
    """
    Turn one record into an OpenAI chat message.

    Strings become plain user messages. PlotRecords become vision messages: the
    chart image plus its caption text when one is already available. When
    ``image_max_px`` is set the chart image is downscaled before encoding.
    """
    if not isinstance(record, PlotRecord):
        return {"role": "user", "content": str(record)}

    content: list[dict] = []
    if record.caption:
        content.append({"type": "text", "text": record.caption})
    plot_path = Path(record.plot_path)
    if plot_path.exists():
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(plot_path, max_px=image_max_px)},
            }
        )
    else:
        logger.warning("Plot image not found, sending caption only: %s", plot_path)
    # Guard against an empty content list (no caption and no image on disk).
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "user", "content": content}


def _chat(
    system_prompt: str,
    record_list: List[Union[PlotRecord, str]],
    *,
    max_tokens: int | None = None,
    choices: List[str] | None = None,
    image_max_px: int | None = None,
) -> str:
    """
    Core chat-completions call shared by the VLM helpers.

    ``choices`` constrains the answer to one of the given strings via vLLM's
    guided decoding (``extra_body={"guided_choice": [...]}``), which both
    guarantees a parseable answer and lets generation stop almost immediately.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        _record_to_message(record, image_max_px=image_max_px) for record in record_list
    )

    kwargs: dict = {
        "model": os.environ.get("STORYBEAR_VLM_MODEL", DEFAULT_MODEL),
        "messages": messages,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if choices is not None:
        kwargs["extra_body"] = {"guided_choice": list(choices)}

    response = _create_with_retry(**kwargs)
    return (response.choices[0].message.content or "").strip()


def it2t_summary_func(
    system_prompt: str,
    record_list: List[Union[PlotRecord, str]],
) -> str:
    """
    Send a system prompt plus a list of records to the remote VLM and return the
    generated text. Mirrors ``local_inference.it2t_summary_func``.
    """
    return _chat(system_prompt, record_list)


def it2t_compare_func(
    system_prompt: str,
    record_list: List[Union[PlotRecord, str]],
) -> str:
    """
    Foodie's pairwise comparison call: constrain the answer to FIRST/SECOND via
    guided decoding and downscale the chart images. Returns exactly "FIRST" or
    "SECOND", so the caller's ``== "FIRST"`` check is reliable.
    """
    compare_px = int(
        os.environ.get("STORYBEAR_COMPARE_IMAGE_PX", DEFAULT_COMPARE_IMAGE_PX)
    )
    return _chat(
        system_prompt,
        record_list,
        max_tokens=8,
        choices=["FIRST", "SECOND"],
        image_max_px=compare_px,
    )


# ---------------------------------------------------------------------------
# Image-to-image (FLUX)
# ---------------------------------------------------------------------------

def flux_i2i_func(prompt: str, file_path: Union[str, Path]) -> Path:
    """
    Send the image at ``file_path`` plus a style ``prompt`` to the remote FLUX
    endpoint and overwrite the file with the edited result. Mirrors
    ``local_inference.flux_i2i_func`` (which edits the image in place).
    """
    import requests

    file_path = Path(file_path)
    save_file_path = file_path.parent / (file_path.stem + "_mod.png")
    from PIL import Image
    img = Image.open(file_path).convert("RGB")
    w, h = klein_size(*img.size)
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    # img.thumbnail((512, 512))
    # img.save(file_path)

    img.save(save_file_path)
    # background = Image.new('RGBA', img.size, (255,255,255))
    # alpha_composite = Image.alpha_composite(background, img)
    # alpha_composite.save(save_file_path)
    # img.save(save_file_path)

    with open(save_file_path, "rb") as f:
        response = requests.post(
            _flux_url(),
            data={"prompt": prompt},
            files={"image": (save_file_path.name, f, "image/png")},
            headers=_proxy_auth_headers(),
            timeout=_timeout(),
        )
    response.raise_for_status()

    with open(save_file_path, "wb") as f:
        f.write(response.content)
    logger.info("Artist: wrote edited image to %s", save_file_path)
    return save_file_path
