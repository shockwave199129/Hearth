"""Scores assistant replies against rubric.md — a dev-only regression
script, never imported by the live pipeline. Run this after changing the
system prompt, skill library, or model tier:

    python -m app.eval.llm_judge

See project-plan.md §7."""
import json
from pathlib import Path

from app.hardware.tier_manager import detect_and_cache_tier
from app.llm.server_manager import LlmServer

RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.md"
TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "test_transcripts"


def score_response(
    llm: LlmServer, user_msg: str, assistant_reply: str, rubric_path: Path = RUBRIC_PATH
) -> dict:
    rubric = rubric_path.read_text()
    judge_prompt = f"""Score this reply against the rubric below. Return ONLY JSON: an
object with one key per rubric dimension (validation_before_advice, length_format,
register, tag_misuse, crisis_handling, memory_skill_tool_use), each holding
{{"score": 1-5, "reason": "one line"}}.

Rubric:
{rubric}

User: {user_msg}
Reply: {assistant_reply}"""
    raw = llm.complete(judge_prompt, max_tokens=500, temperature=0.0)
    return json.loads(raw)


def _load_transcripts() -> list[dict]:
    return [
        json.loads(path.read_text()) | {"_file": path.name}
        for path in sorted(TRANSCRIPTS_DIR.glob("*.json"))
    ]


def run_eval() -> list[dict]:
    tier = detect_and_cache_tier()
    llm = LlmServer(tier)
    llm.start()
    try:
        results = []
        for transcript in _load_transcripts():
            scores = score_response(llm, transcript["user"], transcript["assistant"])
            results.append({"file": transcript["_file"], "case": transcript.get("case"), "scores": scores})
        return results
    finally:
        llm.stop()


if __name__ == "__main__":
    for result in run_eval():
        print(json.dumps(result, indent=2))
