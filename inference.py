"""
Inference Script — Hallucination Detector RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=easy env=hallucination-detector model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=has_hallucination=True,claim=completed in 1902,confidence=1.00 reward=1.00 done=false error=null
    [STEP] step=2 action=has_hallucination=True,claim=New Delhi,confidence=1.00 reward=0.00 done=false error=null
    [END] success=true steps=8 score=1.000 rewards=1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
"""

import asyncio
import os
import json
import time
import csv
from typing import List, Optional

from openai import OpenAI

from client import HallucinationEnvClient
from models import HallucinationAction
from dotenv import load_dotenv
load_dotenv(override=True)
# ── Environment credentials — read from environment variables ─────────
# HF_TOKEN has no default (mandatory, must be set by user)
# API_BASE_URL and MODEL_NAME have defaults per competition requirements
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://rushikeshbathe096-hallucination-detector.hf.space")

# ── Task configuration ────────────────────────────────────────────────
TASKS = ["easy", "medium", "hard", "expert"]
BENCHMARK = "hallucination-detector"
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]


# ── Structured stdout logging — exact format required by competition ───

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line after env.close() — always emitted even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt for the hallucination detection agent ───────────────

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


# ── LLM output parsing ────────────────────────────────────────────────

def parse_llm_output(raw: str) -> HallucinationAction:
    """Parse raw LLM JSON output into a typed HallucinationAction.
    Falls back to safe default (no hallucination, 0.5 confidence) on error.
    """
    try:
        text = raw.strip()
        # Strip markdown code fences if present
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
        # Safe fallback — never crash inference due to parse error
        return HallucinationAction(
            has_hallucination=False,
            hallucinated_claim=None,
            correct_fact=None,
            confidence=0.5
        )


def get_action(client: OpenAI, reference: str, response: str) -> HallucinationAction:
    """Call LLM with the reference and response, return parsed action.
    Uses exponential backoff retry (3 attempts) for rate limits and timeouts.
    """
    user_content = (
        f"REFERENCE DOCUMENT:\n{reference}\n\n"
        f"LLM RESPONSE:\n{response}"
    )

    # Skip seed param for providers that reject it (e.g. Gemini)
    extra_kwargs = {}
    if not any(x in API_BASE_URL for x in ["generativelanguage.googleapis.com", "gemini"]):
        extra_kwargs["seed"] = 42

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content}
                ],
                temperature=0,      # deterministic — required for reproducibility
                max_tokens=256,
                stream=False,
                timeout=45.0,       # increased for slow free-tier providers
                **extra_kwargs
            )
            raw = (completion.choices[0].message.content or "").strip()
            return parse_llm_output(raw)
        except Exception as exc:
            if attempt < max_retries - 1:
                # Exponential backoff: 3s, 5s before final failure
                wait_time = (2 ** attempt) + 1
                print(
                    f"[DEBUG] LLM call failed (attempt {attempt+1}/{max_retries}): {exc}. "
                    f"Retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
            else:
                print(f"[DEBUG] LLM call failed after {max_retries} attempts: {exc}", flush=True)
                # Return safe fallback — do not crash inference
                return HallucinationAction(
                    has_hallucination=False,
                    hallucinated_claim=None,
                    correct_fact=None,
                    confidence=0.5
                )


# ── Task runner — one complete episode ───────────────────────────────

async def run_task(task_id: str, client: OpenAI) -> float:
    """Run one complete task episode via WebSocket EnvClient.
    Returns final episode score in [0, 1].
    Always emits [END] even if an exception occurs.
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Connect to environment via WebSocket — one connection per task
        async with HallucinationEnvClient(base_url=ENV_BASE_URL) as env:

            # Reset environment and get first observation
            result = await env.reset(task_id=task_id)
            obs = result.observation

            step = 0
            while not obs.done:
                step += 1

                # Call LLM to get action for current observation
                action = get_action(
                    client,
                    reference=obs.reference_document,
                    response=obs.llm_response
                )

                # Serialise action for [STEP] log line
                action_str = (
                    f"has_hallucination={action.has_hallucination},"
                    f"claim={action.hallucinated_claim or 'null'},"
                    f"confidence={action.confidence:.2f}"
                )

                # Submit action to environment
                result = await env.step(action)
                obs    = result.observation
                reward = result.reward if result.reward is not None else 0.0
                done   = result.done

                rewards.append(reward)
                steps_taken = step

                # Emit [STEP] immediately after env.step() returns
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                if done:
                    break

            # Extract final episode score from last observation
            score = obs.score if hasattr(obs, "score") else 0.001
            # Clamp strictly within (0, 1) — validator rejects boundary values 0.0 and 1.0
            score = min(max(score, 0.001), 0.999)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        # Always emit [END] — matches sample script finally block pattern
        # env.close() is handled automatically by async context manager
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main entry point ──────────────────────────────────────────────────

async def main() -> None:
    """Run all three tasks sequentially and save results."""

    # Single OpenAI client shared across all tasks — matches sample script
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Optional rate-limit delay between tasks (set REQUEST_DELAY=2 for slow providers)
    request_delay = float(os.getenv("REQUEST_DELAY", "0"))

    results = {}
    for task_id in TASKS:
        score = await run_task(task_id, client)
        results[task_id] = score
        if request_delay > 0:
            time.sleep(request_delay)

    # Print summary
    print("\n" + "=" * 45, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 45, flush=True)
    for task_id, score in results.items():
        print(f"  {task_id:<10} {score:.4f}", flush=True)
    avg = sum(results.values()) / len(results)
    print(f"  {'average':<10} {avg:.4f}", flush=True)

    # Save benchmark results to CSV for reproducibility tracking
    csv_file = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "easy", "medium", "hard", "average"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            MODEL_NAME,
            f"{results.get('easy', 0.0):.4f}",
            f"{results.get('medium', 0.0):.4f}",
            f"{results.get('hard', 0.0):.4f}",
            f"{avg:.4f}"
        ])
    print(f"\n[INFO] Results saved to {csv_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
