# grader.py
import re
from typing import Any, Dict, Tuple


# ── Text utilities ───────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_overlap(a: str, b: str) -> int:
    """Count shared tokens between two strings after normalisation."""
    a_words = set(_normalise(a).split())
    b_words = set(_normalise(b).split())
    # Exclude very common stop words that add noise
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "to", "it", "was", "at"}
    a_words -= stopwords
    b_words -= stopwords
    return len(a_words & b_words)


def _extract_numbers(text: str) -> set:
    """Extract all numeric strings from text (handles commas and decimals)."""
    if not text:
        return set()
    return set(re.findall(r"\d[\d,\.]*", text))


def _phrase_matched(claim: str, phrases: list) -> bool:
    """
    Return True if the agent's claim meaningfully identifies
    a ground truth hallucinated phrase. Uses three strategies:

    Strategy 1 — Exact substring (fast path):
        "1902" matches "completed in 1902"

    Strategy 2 — Keyword overlap >= 2 (handles paraphrasing):
        "the year 1902 is wrong" shares 2 tokens with "completed in 1902"
        after stop word removal: {year, 1902} ∩ {completed, 1902} = {1902} → 1
        "year 1902 Eiffel" shares: {year, 1902, eiffel} ∩ {1902} = {1902} → 1

    Strategy 3 — Numeric match (key for date/number hallucinations):
        If the ground truth phrase contains a specific number and
        the agent's claim contains that same number, count it as a match.
        This handles: "the year is 1902" matching "completed in 1902"
    """
    if not claim or not phrases:
        return False

    claim_norm = _normalise(claim)

    for phrase in phrases:
        phrase_norm = _normalise(phrase)

        # Strategy 1: exact substring either direction
        if claim_norm in phrase_norm or phrase_norm in claim_norm:
            return True

        # Strategy 2: keyword overlap
        if _keyword_overlap(claim, phrase) >= 2:
            return True

        # Strategy 3: numeric match — most important for dates and figures
        phrase_nums = _extract_numbers(phrase)
        claim_nums = _extract_numbers(claim)
        if phrase_nums and claim_nums and (phrase_nums & claim_nums):
            return True

    return False


def _correction_matched(correct_fact: str, corrections: list) -> bool:
    """
    Return True if the agent's correct_fact meaningfully matches
    a ground truth correction. Same three strategies as _phrase_matched.
    """
    if not correct_fact or not corrections:
        return False

    correct_norm = _normalise(correct_fact)

    for correction in corrections:
        correction_norm = _normalise(correction)

        # Strategy 1: exact substring
        if correct_norm in correction_norm or correction_norm in correct_norm:
            return True

        # Strategy 2: keyword overlap
        if _keyword_overlap(correct_fact, correction) >= 2:
            return True

        # Strategy 3: numeric match
        correction_nums = _extract_numbers(correction)
        fact_nums = _extract_numbers(correct_fact)
        if correction_nums and fact_nums and (correction_nums & fact_nums):
            return True

    return False


# ── Main grader ──────────────────────────────────────────────────────

def grade(action, sample: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score an agent's action against a sample's ground truth.

    Returns (final_score: float, feedback: str).
    Score is always in [0.0, 1.0].

    Weight breakdown (sums to 1.0 before calibration):
        Check 1 — hallucination detected:     0.50
        Check 2 — wrong phrase identified:    0.30
        Check 3 — correct fact provided:      0.20
        Confidence calibration:               ±0.10 (does not exceed clamp)

    Check 2 and Check 3 only run if Check 1 passed.
    This prevents partial credit for agents that get the
    fundamental detection wrong.
    """
    score = 0.0
    feedback_parts = []

    gt_has       = sample["ground_truth_has_hallucination"]
    gt_phrases   = sample["ground_truth_hallucinated_phrases"]
    gt_corrections = sample["ground_truth_corrections"]

    # ── Check 1: Hallucination detection (0.50) ──────────────────────
    check1_passed = (action.has_hallucination == gt_has)

    if check1_passed:
        score += 0.50
        if gt_has:
            feedback_parts.append("✓ Hallucination correctly detected.")
        else:
            feedback_parts.append("✓ Clean sample correctly identified — no hallucination.")
    else:
        if gt_has:
            feedback_parts.append("✗ Missed: a hallucination exists but was not detected.")
        else:
            feedback_parts.append("✗ False alarm: no hallucination exists but one was claimed.")

    # ── Check 2: Wrong phrase identified (0.30) ───────────────────────
    # Only runs if Check 1 passed. Applies only to hallucinated samples.
    if check1_passed:
        if gt_has:
            # Sample has real hallucinations — check if agent found one
            if _phrase_matched(action.hallucinated_claim, gt_phrases):
                score += 0.30
                feedback_parts.append("✓ Hallucinated phrase correctly identified.")
            else:
                if action.hallucinated_claim:
                    feedback_parts.append(
                        f"✗ Phrase not matched. "
                        f"Closest expected: '{gt_phrases[0] if gt_phrases else 'N/A'}'"
                    )
                else:
                    feedback_parts.append("✗ No hallucinated phrase provided.")
        else:
            # Clean sample — agent correctly said no hallucination
            # Give full Check 2 credit if agent did not invent a claim
            if not action.hallucinated_claim:
                score += 0.30
                feedback_parts.append("✓ Correctly provided no hallucinated phrase.")
            else:
                feedback_parts.append(
                    "✗ Invented a hallucinated phrase on a clean sample — penalty."
                )
                score -= 0.10  # explicit penalty for false claim on clean sample

    # ── Check 3: Correct fact provided (0.20) ────────────────────────
    # Only runs if Check 1 passed. Applies only to hallucinated samples.
    if check1_passed:
        if gt_has:
            if _correction_matched(action.correct_fact, gt_corrections):
                score += 0.20
                feedback_parts.append("✓ Correct fact matches reference.")
            else:
                if action.correct_fact:
                    feedback_parts.append(
                        f"✗ Correct fact not matched. "
                        f"Expected something close to: '{gt_corrections[0] if gt_corrections else 'N/A'}'"
                    )
                else:
                    feedback_parts.append("✗ No correct fact provided.")
        else:
            # Clean sample — give Check 3 credit if agent did not invent a correction
            if not action.correct_fact:
                score += 0.20
                feedback_parts.append("✓ Correctly provided no correction.")
            else:
                feedback_parts.append(
                    "✗ Invented a correction on a clean sample — penalty."
                )
                score -= 0.05  # small penalty for inventing correction on clean sample

    # ── Confidence calibration (±0.10) ───────────────────────────────
    # Symmetric: rewards confident correct agents, penalises confident wrong agents.
    # Uses same multiplier (0.1) in both directions — mentor's formula.
    confidence = action.confidence if action.confidence is not None else 0.5
    confidence = max(0.0, min(1.0, confidence))

    is_correct = score >= 0.75
    calibration_bonus = 0.1 * (confidence if is_correct else -confidence)
    score += calibration_bonus

    # ── Final clamp ───────────────────────────────────────────────────
    score = round(max(0.0, min(1.0, score)), 4)

    # ── Feedback summary ─────────────────────────────────────────────
    feedback = " | ".join(feedback_parts)
    if score >= 0.90:
        feedback += " || EXCELLENT"
    elif score >= 0.70:
        feedback += " || GOOD"
    elif score >= 0.40:
        feedback += " || PARTIAL"
    else:
        feedback += " || INCORRECT"

    return score, feedback


# ── Self-tests (run: python grader.py) ───────────────────────────────

if __name__ == "__main__":
    from models import HallucinationAction

    hallucinated_sample = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["completed in 1902"],
        "ground_truth_corrections": ["completed in 1889"]
    }

    clean_sample = {
        "ground_truth_has_hallucination": False,
        "ground_truth_hallucinated_phrases": [],
        "ground_truth_corrections": []
    }

    # ── Test 1: Perfect answer on hallucinated sample ────────────────
    perfect = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="completed in 1902",
        correct_fact="completed in 1889",
        confidence=0.9
    )
    s1, f1 = grade(perfect, hallucinated_sample)
    assert s1 >= 0.90, f"FAIL Test 1: expected ≥0.90, got {s1}"
    print(f"Test 1 PASS ({s1}): {f1}")

    # ── Test 2: Wrong detection — missed hallucination ───────────────
    wrong = HallucinationAction(
        has_hallucination=False,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=0.9
    )
    s2, f2 = grade(wrong, hallucinated_sample)
    assert s2 <= 0.15, f"FAIL Test 2: expected ≤0.15, got {s2}"
    print(f"Test 2 PASS ({s2}): {f2}")

    # ── Test 3: Determinism — same input, same output ────────────────
    s3a, _ = grade(perfect, hallucinated_sample)
    s3b, _ = grade(perfect, hallucinated_sample)
    assert s3a == s3b, f"FAIL Test 3: grader is not deterministic ({s3a} != {s3b})"
    print(f"Test 3 PASS: deterministic ({s3a})")

    # ── Test 4: Clean sample correctly identified ────────────────────
    clean_action = HallucinationAction(
        has_hallucination=False,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=0.8
    )
    s4, f4 = grade(clean_action, clean_sample)
    assert s4 >= 0.90, f"FAIL Test 4: expected ≥0.90, got {s4}"
    print(f"Test 4 PASS ({s4}): {f4}")

    # ── Test 5: Paraphrased claim still scores (strategy 2) ──────────
    paraphrase = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="the year 1902 is incorrect",   # paraphrase
        correct_fact="it should be 1889",                  # paraphrase
        confidence=0.8
    )
    s5, f5 = grade(paraphrase, hallucinated_sample)
    assert s5 >= 0.70, f"FAIL Test 5: expected ≥0.70 for paraphrase, got {s5}"
    print(f"Test 5 PASS ({s5}): {f5}")

    # ── Test 6: False alarm on clean sample is penalised ─────────────
    false_alarm = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="completed in 1902",
        correct_fact="completed in 1889",
        confidence=0.9
    )
    s6, f6 = grade(false_alarm, clean_sample)
    assert s6 <= 0.15, f"FAIL Test 6: expected ≤0.15 for false alarm, got {s6}"
    print(f"Test 6 PASS ({s6}): {f6}")

    # ── Test 7: Always-True strategy cannot game the grader ──────────
    always_true = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=1.0
    )
    s7_hall, _ = grade(always_true, hallucinated_sample)   # Check 1 passes
    s7_clean, _ = grade(always_true, clean_sample)          # Check 1 fails
    avg_always_true = (s7_hall + s7_clean) / 2
    assert avg_always_true <= 0.40, f"FAIL Test 7: always-True agent scores too high ({avg_always_true})"
    print(f"Test 7 PASS: always-True average = {avg_always_true} (≤0.40)")

    # ── Test 8: Numeric match works for digit-reversed errors ─────────
    digit_reversed = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="83 percent",   # ground truth has "83 percent"
        correct_fact="38 percent",
        confidence=0.7
    )
    numeric_sample = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["contributed 83 percent of total revenue"],
        "ground_truth_corrections": ["contributed 38 percent of total revenue"]
    }
    s8, f8 = grade(digit_reversed, numeric_sample)
    assert s8 >= 0.70, f"FAIL Test 8: expected ≥0.70 for numeric match, got {s8}"
    print(f"Test 8 PASS ({s8}): {f8}")

    print("\n✓ All 8 grader tests passed.")