# inference.py — root directory, named exactly this
import asyncio
import os
import json
from typing import List, Optional
from openai import OpenAI
from client import HallucinationEnvClient
from models import HallucinationAction

# ── Required env vars (exact names from competition) ─────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "hallucination-detector"
SUCCESS_SCORE_THRESHOLD = 0.5


# ── Exact log format from sample script ──────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


# ── CHANGE 1: Remove score parameter from log_end ────────────────────
# Sample script format: [END] success=true steps=3 rewards=0.00,0.00,1.00
# No score field in the sample script
def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a hallucination detection expert.

You will be given:
1. REFERENCE DOCUMENT — the ground truth facts
2. LLM RESPONSE — which may contain factual errors

Compare them carefully. A hallucination occurs when the LLM response:
- States something that directly contradicts the reference
- Removes a critical qualifier or restriction that changes the meaning
- Swaps numbers, names, or dates to incorrect values

A response is NOT a hallucination if it:
- Simply omits sentences from the reference without changing meaning
- Uses slightly different wording to say the same thing
- Mentions a subset of correct facts from the reference
- States something that is correct according to the reference

Before answering, ask yourself: does the LLM response state anything
that is WRONG according to the reference? If it only says correct things
(even fewer of them), has_hallucination = false.

Respond ONLY with valid JSON — no explanation, no markdown fences:
{
  "has_hallucination": true or false,
  "hallucinated_claim": "the exact wrong phrase from the LLM response, or null",
  "correct_fact": "what the reference document actually says, or null",
  "confidence": 0.0 to 1.0
}

Rules:
- has_hallucination = true ONLY if LLM response directly contradicts reference
- has_hallucination = true if a critical restriction was silently removed
- has_hallucination = false if the response just says fewer correct things
- hallucinated_claim: quote the specific wrong phrase from the LLM response
- correct_fact: quote the correct value from the reference document
- confidence: be honest — subtle errors deserve 0.6 or 0.7, not 1.0
- Return JSON only. Nothing else."""


# ── LLM call ──────────────────────────────────────────────────────────

def parse_llm_output(raw: str) -> HallucinationAction:
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = [l for l in text.splitlines() if not l.startswith("```")]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        return HallucinationAction(
            has_hallucination=bool(data.get("has_hallucination", False)),
            hallucinated_claim=data.get("hallucinated_claim") or None,
            correct_fact=data.get("correct_fact") or None,
            confidence=float(data.get("confidence", 0.5))
        )
    except Exception:
        return HallucinationAction(
            has_hallucination=False,
            hallucinated_claim=None,
            correct_fact=None,
            confidence=0.5
        )


def get_action(client: OpenAI, reference: str, response: str) -> HallucinationAction:
    user_content = (
        f"REFERENCE DOCUMENT:\n{reference}\n\n"
        f"LLM RESPONSE:\n{response}"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            temperature=0,
            max_tokens=256,
            seed=42,        # forces determinism
            stream=False
        )
        raw = (completion.choices[0].message.content or "").strip()
        return parse_llm_output(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return HallucinationAction(
            has_hallucination=False,
            hallucinated_claim=None,
            correct_fact=None,
            confidence=0.5
        )


# ── Task runner ───────────────────────────────────────────────────────

async def run_task(task_id: str, client: OpenAI) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with HallucinationEnvClient(base_url=ENV_BASE_URL) as env:
            result = await env.reset(task_id=task_id)
            obs = result.observation

            step = 0
            while not obs.done:
                step += 1

                action = get_action(
                    client,
                    reference=obs.reference_document,
                    response=obs.llm_response
                )

                action_str = (
                    f"has_hallucination={action.has_hallucination},"
                    f"claim={action.hallucinated_claim or 'null'},"
                    f"confidence={action.confidence:.2f}"
                )

                result = await env.step(action)
                obs    = result.observation
                reward = result.reward if result.reward is not None else 0.0
                done   = result.done

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                if done:
                    break

            score = obs.score if hasattr(obs, "score") else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        # ── CHANGE 2: Always emit [END] even on exception ─────────────
        # Matches sample script finally block pattern exactly
        # No score field — matches sample format
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    for task_id in TASKS:
        score = await run_task(task_id, client)
        results[task_id] = score

    print("\n" + "=" * 45, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 45, flush=True)
    for task_id, score in results.items():
        print(f"  {task_id:<10} {score:.4f}", flush=True)
    avg = sum(results.values()) / len(results)
    print(f"  {'average':<10} {avg:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())