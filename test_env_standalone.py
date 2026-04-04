# save as test_env_standalone.py in project root
# run as: python test_env_standalone.py

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./server")

from server.environment import HallucinationEnvironment
from models import HallucinationAction

env = HallucinationEnvironment()

# ── Test 1: reset returns a valid observation ─────────────────────
obs = env.reset("easy")
assert obs.done == False,              "T1 FAIL: done should be False"
assert obs.sample_index == 0,         "T1 FAIL: index should be 0"
assert obs.total_samples == 8,        "T1 FAIL: easy has 8 samples"
assert len(obs.reference_document) > 0, "T1 FAIL: reference_document is empty"
assert obs.reward is None,            "T1 FAIL: reward should be None on reset"
print(f"T1 PASS — reset() returned sample 0/8")

# ── Test 2: step returns reward and advances index ────────────────
action = HallucinationAction(
    has_hallucination=True,
    hallucinated_claim="completed in 1902",
    correct_fact="completed in 1889",
    confidence=0.8
)
obs2, reward, done, info = env.step(action)
assert isinstance(reward, float),   "T2 FAIL: reward must be float"
assert obs2.steps_taken == 1,       "T2 FAIL: steps_taken should be 1"
assert obs2.sample_index == 1,      "T2 FAIL: index should advance to 1"
assert obs2.done == False,          "T2 FAIL: episode not done after 1 step"
assert "sample_score" in info,      "T2 FAIL: info must contain sample_score"
print(f"T2 PASS — step() returned reward={reward}, score={obs2.score}")

# ── Test 3: state() returns correct values ────────────────────────
state = env.state()
assert state.steps_taken == 1,      "T3 FAIL: state steps_taken should be 1"
assert state.sample_index == 1,     "T3 FAIL: state index should be 1"
assert state.is_done == False,      "T3 FAIL: state is_done should be False"
assert state.episode_id is not None, "T3 FAIL: episode_id should be set"
print(f"T3 PASS — state() returns steps_taken=1, episode_id set")

# ── Test 4: state() before reset returns safe defaults ────────────
fresh = HallucinationEnvironment()
s = fresh.state()
assert s.episode_score == 0.0,  "T4 FAIL: default episode_score should be 0.0"
assert s.task_id == "",          "T4 FAIL: default task_id should be empty string"
assert s.is_done == False,       "T4 FAIL: default is_done should be False"
print(f"T4 PASS — state() before reset returns safe defaults")

# ── Test 5: full easy episode terminates correctly ────────────────
env3 = HallucinationEnvironment()
env3.reset("easy")
steps_taken = 0
done = False
final_obs = None
while not done:
    final_obs, reward, done, _ = env3.step(action)
    steps_taken += 1

assert done == True,              "T5 FAIL: episode should be done"
assert steps_taken == 8,          "T5 FAIL: easy has 8 samples, should take 8 steps"
assert final_obs.done == True,    "T5 FAIL: final obs.done should be True"
assert 0.0 <= final_obs.score <= 1.0, "T5 FAIL: final score out of range"
print(f"T5 PASS — full episode completed in {steps_taken} steps, score={final_obs.score}")

# ── Test 6: calling step after done raises RuntimeError ──────────
error_raised = False
try:
    env3.step(action)
except RuntimeError:
    error_raised = True
assert error_raised, "T6 FAIL: step() after done should raise RuntimeError"
print(f"T6 PASS — step() after done raises RuntimeError correctly")

# ── Test 7: reset() starts a fresh episode ───────────────────────
obs_fresh = env3.reset("medium")
assert obs_fresh.done == False,       "T7 FAIL: done should be False after reset"
assert obs_fresh.sample_index == 0,   "T7 FAIL: index should be 0 after reset"
assert obs_fresh.total_samples == 10, "T7 FAIL: medium has 10 samples"
state_fresh = env3.state()
assert state_fresh.is_done == False,  "T7 FAIL: state is_done should reset"
assert state_fresh.steps_taken == 0,  "T7 FAIL: steps should reset to 0"
print(f"T7 PASS — reset() starts fresh episode on medium task")

print("\n✓ All 7 environment tests passed — ready for integration")