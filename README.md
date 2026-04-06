# Hallucination Detector RL Environment

> **An OpenEnv environment where agents detect factual hallucinations in LLM-generated responses by comparing them against reference documents and submitting structured verification actions.**

LLMs hallucinate. They generate confident, fluent, completely wrong statements. This is the number one unsolved problem in production AI deployment. Every company using AI faces this daily. No RL environment existed to train agents to detect it. **We built the first one.**

---

## Why This Matters

Hallucinations reduce trust in AI systems across products serving billions of users. Incorrect but **confident** outputs are more dangerous than uncertain ones — a user cannot tell the difference.

An agent trained in this environment learns not just to detect errors, but to express **calibrated uncertainty** — knowing when to be confident and when to hedge. This is a core requirement for safe AI deployment at scale.

---

## OpenEnv Compliance

Built on the official OpenEnv framework:

| Requirement | Status |
|---|---|
| Inherits from `openenv.core.Environment` | ✅ |
| Served via `create_web_interface_app` | ✅ |
| Compatible with `EnvClient` WebSocket | ✅ |
| `openenv validate --verbose` passes | ✅ |
| `SUPPORTS_CONCURRENT_SESSIONS = True` | ✅ |
| Tagged with `openenv` on HF Spaces | ✅ |

---

## How It Works
Reference Document (ground truth)
+
LLM Response (may contain errors)
↓
Agent reads both
↓
Agent submits structured action
↓
Deterministic grader scores 4 dimensions
↓
Environment returns delta reward + feedback + next sample
↓
Repeat until episode done

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `has_hallucination` | `bool` | True if LLM response contradicts reference |
| `hallucinated_claim` | `str \| null` | Exact wrong phrase from the LLM response |
| `correct_fact` | `str \| null` | What the reference document actually says |
| `confidence` | `float 0–1` | Agent confidence in its answer |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `reference_document` | `str` | Ground truth paragraph |
| `llm_response` | `str` | Response that may contain errors |
| `done` | `bool` | True when episode ends |
| `reward` | `float` | Delta reward for this step |
| `score` | `float` | Running episode average |
| `feedback` | `str` | Grader explanation with partial credit breakdown |
| `sample_index` | `int` | Current sample number |
| `total_samples` | `int` | Total samples in this task |
| `steps_taken` | `int` | Steps completed so far |
| `max_steps` | `int` | Maximum steps allowed |

---

## Tasks

| Task | Samples | Design | Max Steps |
|---|---|---|---|
| `easy` | 8 | One obvious error per sample — wrong year, name, or city. 2 clean samples. | 10 |
| `medium` | 10 | Two to three mixed errors — digit swaps, wrong attribution, false facts. 2 clean samples. | 12 |
| `hard` | 15 | Negation traps, entity flipping, adversarial clean samples, multi-hop reasoning errors. 3 clean samples. | 15 |

### Hard Task Design — What Makes It Genuinely Difficult

The hard task contains samples specifically designed to fool capable LLMs:

- **Negation Traps** — "not liable unless X" becomes "liable even without X"
- **Entity Role Confusion** — who acquired whom gets reversed
- **Adversarial Clean Samples** — counter-intuitive true facts that sound wrong (Venus day > Venus year)
- **Temporal Confusion** — Oort Cloud vs interstellar space boundary
- **Unit Shifting** — 384,400 kilometres vs 384,400 metres
- **Partial Truth** — drug mechanism is half right and half completely wrong

---

## Reward Function

Reward is **delta-based** — the change in running episode average:
step 1:  reward = sample_score
step N:  reward = sample_score - average(scores[0..N-1])

This provides a genuine learning signal at every step. Agents receive immediate feedback on whether each detection **improved or degraded** their overall performance — not just a binary end-of-episode signal.

---

## Scoring Breakdown

| Component | Weight | Description |
|---|---|---|
| Hallucination detection | **0.50** | Correctly identifies whether hallucination exists |
| Phrase identification | **0.30** | Identifies the specific wrong phrase (fuzzy matched) |
| Correct fact | **0.20** | Provides the correct fact from reference |
| Confidence calibration | **±0.10** | Rewards honest confidence, penalises overconfidence |

### Fuzzy Matching — Why It Matters

Agents should not fail because they paraphrase slightly. The grader uses four matching strategies:

1. **Substring containment** — "1902" matches "completed in 1902"
2. **Keyword overlap** — shared content words after stopword removal
3. **Numeric token matching** — "83 percent" matches "contributed 83 percent of total revenue"
4. **Character n-gram similarity** — Jaccard trigrams handle minor typos

---

## Exploit Resistance

| Strategy | Easy | Medium | Hard |
|---|---|---|---|
| Always-True agent | 0.300 | 0.320 | 0.320 |
| Always-False agent | 0.250 | 0.200 | 0.200 |
| Random agent | 0.573 | 0.544 | 0.544 |
| **Correct calibrated agent** | **1.000** | **0.995** | **0.849** |

No lazy strategy scores above 0.60. Only a correct and calibrated agent approaches 1.0.

---

## Baseline Results

Evaluated with `llama-3.1-8b-instant` via Groq API. `temperature=0`, `seed=42`.

| Task | Score | Samples | Notes |
|---|---|---|---|
| easy | **1.000** | 8 | All errors detected correctly |
| medium | **0.995** | 10 | Near-perfect on mixed errors |
| hard | **0.849** | 15 | Adversarial clean samples cause false alarms |
| **average** | **0.948** | 33 | |

Run time: **2 minutes 28 seconds** (limit: 20 minutes)

---

## Project Structure
mini_rl_environment/
├── models.py              # Pydantic typed models — shared contract
├── tasks.py               # 33 samples across 3 difficulty levels
├── grader.py              # Deterministic scoring with fuzzy matching
├── client.py              # WebSocket EnvClient wrapper
├── inference.py           # Baseline agent — reads API creds from env vars
├── openenv.yaml           # Environment manifest
├── Dockerfile             # Container — builds in ~60 seconds
├── README.md
├── tests/
│   └── evaluate_scores.py # Validation — all sections must pass
└── server/
├── init.py
├── app.py             # FastAPI server via OpenEnv factory
├── environment.py     # reset(), step(), state()
└── requirements.txt

---

## Quick Start — Local
```bash
# Install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Verify
curl localhost:7860/health
# → {"status":"ok"}

curl -X POST localhost:7860/reset \
	-H "Content-Type: application/json" \
	-d '{"task_id": "easy"}'
# → JSON with reference_document and llm_response
```

---

## Quick Start — Docker
```bash
docker build -t hallucination-detector .
docker run -p 7860:7860 hallucination-detector
curl localhost:7860/health
```

---

## Run Inference
```bash
# Set credentials
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=your_api_key
export ENV_BASE_URL=http://localhost:7860

# Run
python inference.py
```

**Expected output:**
[START] task=easy env=hallucination-detector model=llama-3.1-8b-instant
[STEP] step=1 action=has_hallucination=True,claim=completed in 1902,confidence=1.00 reward=1.00 done=false error=null
[STEP] step=2 action=has_hallucination=True,claim=New Delhi, India,confidence=1.00 reward=0.00 done=false error=null
...
[END] success=true steps=8 rewards=1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00

---

## Run Validation
```bash
python grader.py              # 10 unit tests — all must pass
python -m tests.evaluate_scores  # 4 sections — all must pass
```

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama-3.1-8b-instant` |
| `HF_TOKEN` | API key | `gsk_...` |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — returns `{"status":"ok"}` |
| `/reset` | POST | Start new episode, returns first observation |
| `/step` | POST | Submit action, returns next observation and reward |
| `/state` | GET | Current episode metadata |

---

## What Makes This Stand Out

- **Novel domain** — first OpenEnv environment for hallucination detection
- **Deterministic grading** — no LLM judge, fully reproducible, same input always same output
- **Real-world relevance** — hallucination is the #1 unsolved problem in production AI
- **Confidence calibration** — trains agents to know when they are uncertain
- **Exploit-resistant** — lazy agents cannot game the grader
- **Delta reward** — genuine learning signal at every step, not sparse end-of-episode reward
- **Adversarial clean samples** — tests false positive rate, not just recall
