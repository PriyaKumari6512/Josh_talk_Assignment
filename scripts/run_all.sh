#!/bin/bash
# scripts/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline for Question 1.
# Run from the project root: bash scripts/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit immediately on any error

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Josh Talks – Hindi ASR Fine-Tuning Pipeline         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Step 1 – Download
echo "▶ Step 1: Downloading audio + transcripts"
python src/data/download_data.py

# Step 2 – Preprocess
echo ""
echo "▶ Step 2: Segmenting audio and building manifests"
python src/data/preprocess.py

# Step 3 – FLEURS
echo ""
echo "▶ Step 3: Preparing FLEURS Hindi test set"
python src/data/prepare_fleurs.py

# Step 4 – Training
echo ""
echo "▶ Step 4: Fine-tuning Whisper-small (this will take a while)"
python src/training/train_whisper.py

# Step 5 – Evaluation
echo ""
echo "▶ Step 5: Evaluating on FLEURS (baseline + fine-tuned)"
python src/evaluation/evaluate_wer.py

# Step 6 – Error analysis
echo ""
echo "▶ Step 6: Running error analysis"
python src/evaluation/error_analysis.py

# Step 7 – Apply fixes
echo ""
echo "▶ Step 7: Applying fix and showing before/after"
python src/evaluation/apply_fixes.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  DONE!  Check outputs/ for results.              ║"
echo "╚══════════════════════════════════════════════════════╝"
