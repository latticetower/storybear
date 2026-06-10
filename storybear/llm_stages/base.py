
import logging
from pathlib import Path
import json
import pandas as pd
from typing import Union, Dict, List

from typing import Any

from storybear.data_structures import PlotRecord, ReportRecord
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Shared LLM mixin
# ---------------------------------------------------------------------------


class _LLMMixin:
    """
    Minimal interface for LLM-powered stages.

    Replace `_call_llm` with your preferred client (Anthropic, OpenAI, etc.).
    For vision-capable stages override `_call_llm_vision` as well.
    """

    SYSTEM_PROMPT: str = "You are a helpful data analysis assistant."

    def _call_llm(self, prompt: str, records_list: List[PlotRecord]) -> str:
        """
        Send *prompt* to the LLM and return the plain-text response.

        TODO: implement with your LLM client, e.g.::

            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        """
        raise NotImplementedError("Override _call_llm() with your LLM client.")

    def _call_llm_vision(self, prompt: str, image_path: Path) -> str:
        """
        Send *prompt* + the image at *image_path* to a vision-capable LLM.

        TODO: implement with your LLM client, e.g.::

            with open(image_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            msg = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": "image/png",
                                    "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return msg.content[0].text
        """
        raise NotImplementedError("Override _call_llm_vision() with your LLM client.")
    
    def _call_llm_image2image(self, prompt: str, image_path: Path) -> Path:
        raise NotImplementedError("Override _call_llm_image2image() with your LLM client.")
    
    def __call__(self, report: ReportRecord) -> ReportRecord:
        raise NotImplementedError("Override __call__() with your LLM client.")

    @staticmethod
    def _parse_json(text: Union[str, Dict]) -> Any:
        """Strip markdown fences and parse JSON from an LLM response."""
        if isinstance(text, dict):
            return text
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
