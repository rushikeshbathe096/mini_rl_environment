# grader.py
import re
from typing import Any, Dict, List, Tuple


# ── Text utilities ───────────────────────────────────────────────────

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", 
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", 
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", 
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", 
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000"
}

def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _word_to_digit(text: str) -> str:
    """Converts written numbers to digits for robust matching."""
    words = text.split()
    return " ".join([WORD_TO_DIGIT.get(w, w) for w in words])

def _keyword_overlap(a: str, b: str) -> int:
    """Count shared tokens between two strings after normalisation."""
    a_words = set(_word_to_digit(_normalise(a)).split())
    b_words = set(_word_to_digit(_normalise(b)).split())
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "to", "it", "was", "at"}
    return len((a_words - stopwords) & (b_words - stopwords))

def _ngram_overlap(s1: str, s2: str, n: int = 3) -> float:
    """Jaccard similarity using character n-grams to defeat minor typos/synonyms."""
    s1 = _word_to_digit(_normalise(s1)).replace(" ", "")
    s2 = _word_to_digit(_normalise(s2)).replace(" ", "")
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    ngrams1 = set([s1[i:i+n] for i in range(len(s1)-n+1)])
    ngrams2 = set([s2[i:i+n] for i in range(len(s2)-n+1)])
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0

def _extract_numbers(text: str) -> set:
    """Extract all numeric strings from text (after word-to-digit conversion)."""
    if not text:
        return set()
    return set(re.findall(r"\d[\d,\.]*", _word_to_digit(text)))

def _is_meaningful_match(claim: str, ground_truths: list) -> bool:
    if not claim or not ground_truths:
        return False
    claim_norm = _word_to_digit(_normalise(claim))
    for t in ground_truths:
        t_norm = _word_to_digit(_normalise(t))
        if claim_norm in t_norm or t_norm in claim_norm:
            return True
        if _keyword_overlap(claim_norm, t_norm) >= 2:
            return True
        if _extract_numbers(claim_norm) and _extract_numbers(t_norm) and (_extract_numbers(claim_norm) & _extract_numbers(t_norm)):
            return True
        if _ngram_overlap(claim_norm, t_norm) >= 0.4:
            return True
    return False

def _phrase_matched(claim: str, phrases: list) -> bool:
    return _is_meaningful_match(claim, phrases)

def _correction_matched(correct_fact: str, corrections: list) -> bool:
    return _is_meaningful_match(correct_fact, corrections)


# ── Coverage scoring for multi-error samples ─────────────────────────

def _count_covered_errors(claim: str, gt_phrases: List[str]) -> int:
    """Count how many distinct ground-truth phrases the agent's claim covers."""
    if not claim or not gt_phrases:
        return 0
    covered = 0
    for phrase in gt_phrases:
        if _is_meaningful_match(claim, [phrase]):
            covered += 1
    return covered


# ── Main grader ──────────────────────────────────────────────────────

def grade(action, sample: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score an agent's action against a sample's ground truth.

    Returns (final_score: float, feedback: str).

    Scoring logic:
        - Detection (has_hallucination correct):       0.30 base
        - Phrase coverage (% of GT phrases matched):   0.40 × coverage_ratio
        - Correction provided:                         0.30 × correction match
        - Confidence calibration:                      multiply by confidence

    Multi-error penalty:
        If sample has N ground-truth errors and the agent only identifies K < N,
        the phrase score is scaled by K/N, enforcing exhaustive detection.

    Anti-cheating:
        False alarm on clean sample → -1.0 × confidence
    """
    score = 0.0
    feedback_parts = []

    gt_has         = sample["ground_truth_has_hallucination"]
    gt_phrases     = sample["ground_truth_hallucinated_phrases"]
    gt_corrections = sample["ground_truth_corrections"]

    agent_has  = action.has_hallucination
    confidence = action.confidence if action.confidence is not None else 0.5
    confidence = max(0.0, min(1.0, confidence))

    # ── Anti-cheating: False alarm on clean text ─────────────────────
    if agent_has and not gt_has:
        score = -1.0 * confidence
        feedback_parts.append("✗ Crucial Error: False alarm on a clean text. Severe Penalty applied.")
    
    # ── Missed hallucination ─────────────────────────────────────────
    elif not agent_has and gt_has:
        score = 0.0
        feedback_parts.append("✗ Missed: A hallucination exists but was not detected.")
    
    # ── Correctly verified clean sample ──────────────────────────────
    elif not agent_has and not gt_has:
        score = 1.0 * confidence
        feedback_parts.append("✓ Clean sample accurately verified.")
    
    # ── Hallucinated sample correctly identified ─────────────────────
    else:  # agent_has and gt_has
        # Base score for correct detection
        score = 0.30
        feedback_parts.append("✓ Hallucination correctly detected.")
        
        # ── Phrase coverage (multi-error aware) ──────────────────────
        num_gt_errors = max(len(gt_phrases), 1)
        covered = _count_covered_errors(action.hallucinated_claim, gt_phrases)
        coverage_ratio = min(covered / num_gt_errors, 1.0)
        
        if covered > 0:
            phrase_score = 0.40 * coverage_ratio
            score += phrase_score
            if covered < num_gt_errors:
                feedback_parts.append(
                    f"⚠ Partial phrase match: found {covered}/{num_gt_errors} errors."
                )
            else:
                feedback_parts.append("✓ All phrases matched.")
        else:
            feedback_parts.append(
                f"✗ Phrase mismatch. Expected: '{gt_phrases[0] if gt_phrases else 'N/A'}'"
            )
            
        # ── Correction quality ───────────────────────────────────────
        if _correction_matched(action.correct_fact, gt_corrections):
            score += 0.30
            feedback_parts.append("✓ Fact matched.")
        else:
            feedback_parts.append(
                f"✗ Fact mismatch. Expected: '{gt_corrections[0] if gt_corrections else 'N/A'}'"
            )
            
        # ── Confidence calibration ───────────────────────────────────
        score = score * confidence

    # Final clamp
    score = round(max(-1.0, min(1.0, score)), 4)

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

    multi_error_sample = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["28 member states", "19 countries"],
        "ground_truth_corrections": ["27 member states", "20 countries"]
    }

    # ── Test 1: Perfect answer on hallucinated sample ────────────────
    perfect = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="completed in 1902",
        correct_fact="completed in 1889",
        confidence=1.0
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
        confidence=1.0
    )
    s4, f4 = grade(clean_action, clean_sample)
    assert s4 >= 0.90, f"FAIL Test 4: expected ≥0.90, got {s4}"
    print(f"Test 4 PASS ({s4}): {f4}")

    # ── Test 5: Paraphrased claim still scores ───────────────────────
    paraphrase = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="the year 1902 is incorrect",
        correct_fact="it should be 1889",
        confidence=1.0
    )
    s5, f5 = grade(paraphrase, hallucinated_sample)
    assert s5 >= 0.90, f"FAIL Test 5: expected ≥0.90, got {s5}"
    print(f"Test 5 PASS ({s5}): {f5}")

    # ── Test 6: False alarm on clean sample is penalised ─────────────
    false_alarm = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="completed in 1902",
        correct_fact="completed in 1889",
        confidence=1.0
    )
    s6, f6 = grade(false_alarm, clean_sample)
    assert s6 <= -0.90, f"FAIL Test 6: expected ≤-0.90, got {s6}"
    print(f"Test 6 PASS ({s6}): {f6}")

    # ── Test 7: Always-True strategy cannot game the grader ──────────
    always_true = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=1.0
    )
    s7_hall, _ = grade(always_true, hallucinated_sample)   # 0.30 (detection only)
    s7_clean, _ = grade(always_true, clean_sample)          # -1.0
    avg_always_true = (s7_hall + s7_clean) / 2
    assert avg_always_true <= 0.0, f"FAIL Test 7: always-True agent scores too high ({avg_always_true})"
    print(f"Test 7 PASS: always-True average = {avg_always_true} (≤0.0)")

    # ── Test 8: Numeric match works for digit-reversed errors ─────────
    digit_reversed = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="83 percent",
        correct_fact="38 percent",
        confidence=1.0
    )
    numeric_sample = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["contributed 83 percent of total revenue"],
        "ground_truth_corrections": ["contributed 38 percent of total revenue"]
    }
    s8, f8 = grade(digit_reversed, numeric_sample)
    assert s8 >= 0.60, f"FAIL Test 8: expected ≥0.60 for numeric match, got {s8}"
    print(f"Test 8 PASS ({s8}): {f8}")

    # ── Test 9: Multi-error — catching only 1 of 2 errors ────────────
    partial_detection = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="28 member states",
        correct_fact="27 member states",
        confidence=1.0
    )
    s9, f9 = grade(partial_detection, multi_error_sample)
    # Should get: 0.30 (detection) + 0.40*(1/2) (partial phrase) + 0.30 (correction) = 0.80
    assert s9 <= 0.85, f"FAIL Test 9: partial detection should be ≤0.85, got {s9}"
    assert s9 >= 0.50, f"FAIL Test 9: partial detection should be ≥0.50, got {s9}"
    print(f"Test 9 PASS ({s9}): {f9}")

    # ── Test 10: Multi-error — catching 0 of 2 errors ────────────────
    no_phrase_match = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="something completely wrong",
        correct_fact="also wrong",
        confidence=1.0
    )
    s10, f10 = grade(no_phrase_match, multi_error_sample)
    # Should get: 0.30 (detection only, no phrase or correction match)
    assert s10 <= 0.35, f"FAIL Test 10: no-match should be ≤0.35, got {s10}"
    print(f"Test 10 PASS ({s10}): {f10}")

    print("\n✓ All 10 grader tests passed.")