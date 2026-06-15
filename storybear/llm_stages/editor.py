
import json
from typing import Tuple, List
from pathlib import Path
import logging
from tqdm.auto import tqdm

from storybear.data_structures import PlotRecord, ReportRecord
from storybear.llm_stages.base import _LLMMixin

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# 6. Editor
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Secretary
# ---------------------------------------------------------------------------
# TODO: fix this

class Secretary:
    """
    Step 5 — Selects the top-N ranked records.

    No LLM involvement; purely deterministic.

    Parameters
    ----------
    top_n:
        Number of (P, C, R) triplets to keep.
    """

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def __call__(self, report: ReportRecord) -> ReportRecord:
        return self.select(report)

    def select(self, report: ReportRecord) -> ReportRecord:
        """Return the top-N records sorted by ranking descending."""
        sorted_records = sorted(report.plot_record_list, key=lambda r: r.ranking, reverse=True)
        selected = sorted_records[: self.top_n]
        logger.info(
            "Secretary: kept %d/%d records (scores: %s)",
            len(selected),
            len(report.plot_record_list),
            [round(r.ranking, 2) for r in selected],
        )
        new_report = ReportRecord(report.header, report.lead, selected)
        return new_report


class Editor(_LLMMixin):
    """
    Step 6 — Produces the report header H and lead paragraph L.

    The Editor sees all selected (P, C, R) triplets and synthesises them
    into a coherent editorial intro.

    Parameters
    ----------
    exaggeration:
        Float in [0, 1].  0 = sober/factual headline; 1 = tabloid-style.
        Passed as instruction to the LLM.
    """

    SYSTEM_PROMPT = (
        "You are smart and talkative squirrel living in a magical forest, who happens also to be an editor-in-chief."
        "You are working on popular science articles for a wide forest community. You are crafting the insights report, "
        "trying to be both easy-to-read and informative. Your inner dream is to become an influencer, that's why you are interested "
        "in the most sensational facts and don't mention exact numbers in your texts.\n"
        "Remember that you are still a squirrel and sometimes you talk about nuts or mention your fluffy tail."
    )

    def __init__(self, exaggeration: float = 0.6) -> None:
        if not 0.0 <= exaggeration <= 1.0:
            raise ValueError("exaggeration must be in [0, 1]")
        self.exaggeration = exaggeration
        self._llm_image2text_func = None

    def __call__(self, report: ReportRecord) -> ReportRecord:
        return self.compose(report)

    def set_llm_image2text(self, it2t_func):
        self._llm_image2text_func = it2t_func

    def compose(self, report: ReportRecord) -> ReportRecord:
        """Generate header H and lead L from the top-N records."""
        if len(report.plot_record_list) == 0:
            raw_header = "The Editor lost his job"
            raw_header = "There is no plots left to describe"
            raw_discussion = "Data was probably stolen by squirrels!"
            return ReportRecord(raw_header, raw_lead, report.plot_record_list, raw_discussion)
        header_prompt, lead_prompt, discussion_prompt = self._build_prompt(report.plot_record_list)
        #prepared_records = [rec.caption for rec in records_list]
        #prepared_records = prepared_records[:5] # TODO: add view
        pbar = tqdm(total=100)
        raw_lead = self._call_llm(lead_prompt, report.plot_record_list)
        print(raw_lead)
        pbar.update(30)
        raw_header = self._call_llm(header_prompt, [raw_lead])
        raw_header = raw_header.strip().split("\n")[0].strip()
        pbar.update(30)
        raw_discussion = self._call_llm(discussion_prompt, [raw_header, raw_lead] + report.plot_record_list)
        pbar.update(40)
        pbar.close()
        print("raw header", raw_header)
        print("raw lead", raw_lead)
        #header = self._parse_response(raw_header)
        #lead = self._parse_response(raw_lead)
        # header, lead = self._parse_response(raw)
        return ReportRecord(raw_header, raw_lead, report.plot_record_list, raw_discussion)

    # TODO: make something with _build_prompt: currently not in use
    def _build_prompt(self, records: list[PlotRecord]) -> str:
        # summaries = "\n\n".join(
        #     f"[{i+1}] Score {r.ranking:.1f} | Columns: {r.columns}\n"
        #     f"Caption: {r.caption}"
        #     for i, r in enumerate(records)
        # )
        exagg_instruction = (
            "Be strictly factual and measured."
            if self.exaggeration < 0.2
            else f"Use a sensationalism level of {self.exaggeration:.0%} "
                 "(0% = dry academic, 100% = clickbait tabloid)."
        )
        header_prompt = (
            f"{self.SYSTEM_PROMPT}\n"
            f"You have the lead describing the data that you have.\n"
            f"Write a short and catchy report HEADER (one punchy title, 1 sentence, 10 words or less)."
            f"{exagg_instruction}"
            #'Return ONLY text.'
        )  #f"{summaries}\n\n"
        lead_prompt = (
            f"{self.SYSTEM_PROMPT}\n"
            f"You have {len(records)} data findings describing the data you have.\n"
            f"Write a report LEAD paragraph "
            f"(1-2 sentences) that captures the most important insight. Don't provide too many details. \n"
            "Possible openings for lead paragraph: 'Scientist discovered that...', 'Data indicates that...', "
            "'In a wast world...', 'Someone must be nuts...'"
            "''"
            f"{exagg_instruction}\n"
            # 'Return ONLY text'
        )

        discussion_prompt = (
            f"{self.SYSTEM_PROMPT}\n"
            f"You have header, lead and {len(records)} data findings describing the data that you have.\n"
            f"Write a discussion raising open questions and concerns related to the problem described in the header."
            "How it is covered in the data? What are the perspectives?"
            "You response should have the following structure:\n"
            "1. One sentence desribing the main problem (which is highlighted in the header).\n"
            "2. For each of data findings: 1-2 short sentences describing how this problem is related to it.\n"
            "3. One sentence: what is the problem that we face and what the squirrel society should do to fix this problem?\n"
            #'Return ONLY text.'
        )  #f"{summaries}\n\n"
        return header_prompt, lead_prompt, discussion_prompt

    def _parse_response(self, raw: str) -> Tuple[str, str]:
        try:
            data = self._parse_json(raw)
            choice = data['choices'][0]
            return choice['message']['content']
            # return data["header"], data["lead"]
        except Exception as exc:
            logger.error("Editor could not parse response: %s — %s", raw[:120], exc)
            return "Data Analysis Report", raw.strip()

    #def _call_llm(self, prompt: str) -> str:
    #    return json.dumps({"header": "New data insights", "lead": "You won't believe to our most recent findings"})
    
    def _call_llm(self, prompt: str, record_list: List[PlotRecord | str]) -> str:
        if self._llm_image2text_func is None:
            return "New data insights from Editor. You won't believe to our most recent findings!"
        # user_messages = [{"role": "user", "content": text} for text in records_list]
        # response_format = {
        #     "type": "json_object",
        #     "schema": {
        #         "type": "object",
        #         "properties": {"header": {"type": "string"}, "lead": {"type": "string"}},
        #         "required": ["header", "lead"],
        #     }
        # }
        res = self._llm_image2text_func(prompt, record_list)
        return res