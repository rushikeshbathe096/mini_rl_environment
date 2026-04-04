# Hallucination Detection RL Environment (Meta x Hugging Face Hackathon)

This project provides a robust Reinforcement Learning (RL) environment to evaluate AI agents on their ability to detect and correct hallucinations in LLM-generated text.

## 🚀 Judge's Quick Start (3-Minute Setup)

### 1. Installation
```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Verify Logic (Instant)
Run the internal grader tests to verify the robust matching and anti-cheating math.
```bash
python grader.py
```
*   **What to look for:** Proof that the environment handles word-to-digit conversion (e.g., "twenty" → 20) and penalizes false alarms (-1.0).

### 3. Launch the Web Dashboard
```bash
python server/app.py
```
*   **Access UI:** Open `http://localhost:7860/web/`
*   **Custom Tab:** Click the **"Custom"** tab to access the **Task Selector** (switch between Easy, Medium, and Hard) and the **Competition Dashboard**.

### 4. Run an RL Agent Loop
To see a live agent (Qwen-72B) solving the environment:
1.  Copy `.env.example` to `.env`.
2.  Add your `HF_TOKEN`.
3.  Run the inference script:
```bash
python inference.py
```

## 🛠 Features

### Robust Grading System
-   **N-Gram Similarity**: Uses Jaccard character 3-grams to handle typos and paraphrasing.
-   **Numeric Normalization**: Converts written number words to digits during evaluation.
-   **Anti-Cheating Protocol**: Built-in penalties (-1.0) for agents that "cry wolf" on clean text.

### High-Quality Hard Dataset
-   **Negation Traps**: Subtle "not" injections.
-   **Entity Flipping**: Switching owners/actors in a sentence.
-   **Unit Shifting**: Changing km to miles while keeping digits.

## 🐳 Docker Support
Build and run locally to simulate Hugging Face Spaces deployment:
```bash
docker build -t hallucination-env .
docker run -p 7860:7860 hallucination-env
```
