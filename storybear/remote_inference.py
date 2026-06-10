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

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "minicpm-v"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 600.0


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _timeout() -> float:
    return float(os.environ.get("STORYBEAR_HTTP_TIMEOUT", DEFAULT_TIMEOUT))


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


def _flux_url() -> str:
    url = os.environ.get("STORYBEAR_FLUX_URL")
    if not url:
        raise RuntimeError(
            "STORYBEAR_FLUX_URL is not set. Point it at the deployed FLUX endpoint, "
            "e.g. https://<workspace>--storybear-flux-klein-web.modal.run/edit"
        )
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
        )
    return _client


# ---------------------------------------------------------------------------
# Image-to-text (VLM)
# ---------------------------------------------------------------------------

def _image_data_url(path: Path) -> str:
    """Encode an image file as an OpenAI-style base64 data URL."""
    suffix = path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in ("jpg", "jpeg") else (suffix or "png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{media_type};base64,{data}"


def _record_to_message(record: Union[PlotRecord, str]) -> dict:
    """
    Turn one record into an OpenAI chat message.

    Strings become plain user messages. PlotRecords become vision messages: the
    chart image plus its caption text when one is already available.
    """
    if not isinstance(record, PlotRecord):
        return {"role": "user", "content": str(record)}

    content: list[dict] = []
    if record.caption:
        content.append({"type": "text", "text": record.caption})
    plot_path = Path(record.plot_path)
    if plot_path.exists():
        content.append(
            {"type": "image_url", "image_url": {"url": _image_data_url(plot_path)}}
        )
    else:
        logger.warning("Plot image not found, sending caption only: %s", plot_path)
    # Guard against an empty content list (no caption and no image on disk).
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "user", "content": content}


def it2t_summary_func(
    system_prompt: str,
    record_list: List[Union[PlotRecord, str]],
) -> str:
    """
    Send a system prompt plus a list of records to the remote VLM and return the
    generated text. Mirrors ``local_inference.it2t_summary_func``.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_record_to_message(record) for record in record_list)

    response = _get_client().chat.completions.create(
        model=os.environ.get("STORYBEAR_VLM_MODEL", DEFAULT_MODEL),
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


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
    with open(file_path, "rb") as f:
        response = requests.post(
            _flux_url(),
            data={"prompt": prompt},
            files={"image": (file_path.name, f, "image/png")},
            timeout=_timeout(),
        )
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)
    logger.info("Artist: wrote edited image to %s", file_path)
    return file_path
